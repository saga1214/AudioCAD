# Copyright 2025 Xiaomi Corporation.
import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, cast, Tuple
import types

import torch
import torch.distributed as dist
from torch import nn
from transformers import StoppingCriteria
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateOutput,
    GenerationConfig,
    StoppingCriteriaList,
    is_deepspeed_zero3_enabled,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2PreTrainedModel,
)
from transformers.utils import is_torchdynamo_compiling

logger = logging.getLogger(__name__)


class MiMoStopper(StoppingCriteria):
    def __init__(
        self,
        group_size: int,
        audio_channels: int,
        stop_tokens: list[int] | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.audio_channels = audio_channels
        self.step = (audio_channels + 1) * group_size

        self.stop_token_ids = set(stop_tokens or [])

        self.max_length = max_length
        self.min_length = min_length or 0

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor):
        is_done = False
        cur_len = input_ids.shape[-1] // self.step
        
        if self.max_length:
            is_done |= cur_len >= self.max_length
            
        if (self.stop_token_ids and 
            input_ids.shape[1] >= self.step and 
            cur_len >= self.min_length):
            last_token = input_ids[0, -self.step].item()
            is_done |= last_token in self.stop_token_ids

        return torch.full(
            (input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool
        )


@dataclass
class MiMoSampler:
    do_sample: bool | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    def process(self, scores: torch.Tensor):
        if self.temperature is not None:
            scores = scores / self.temperature

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, scores.shape[-1])
            indices_to_remove = scores < torch.topk(scores, top_k)[0][:, -1]
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        if self.top_p is not None and 0.0 < self.top_p <= 1.0:
            top_p = self.top_p if 0.0 < self.top_p <= 1.0 else 1.0
            sorted_logits, sorted_indices = torch.sort(scores)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[:, -1] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        return scores

    def process_filter(self, cond_scores: torch.Tensor, guided_scores: torch.Tensor, sample_from: str = "cond_logit"):
        assert sample_from in ["cond_logit",
                               "guided_logit"], f"{sample_from} should be one of ['cond_logit', 'guided_logit']"

        if sample_from == "cond_logit":
            sampling_scores = cond_scores
            filter_scores = guided_scores
        else:
            sampling_scores = guided_scores
            filter_scores = cond_scores

        if self.temperature is not None:
            sampling_scores = sampling_scores / self.temperature
            filter_scores = filter_scores / self.temperature

        # Build the keep/drop mask from filter_scores (False = keep, True = drop).
        mask = torch.zeros_like(filter_scores, dtype=torch.bool)

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, filter_scores.shape[-1])
            top_k_values, _ = torch.topk(filter_scores, top_k)
            mask |= filter_scores < top_k_values[:, -1, None]

        if self.top_p is not None and 0.0 < self.top_p <= 1.0:
            sorted_logits, sorted_indices = torch.sort(filter_scores, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            mask |= indices_to_remove

        final_scores = sampling_scores.masked_fill(mask, float("-inf"))

        return final_scores

    def sample(self, scores: torch.Tensor, removed_tokens: list[int] | None = None, auxiliary_scores: torch.Tensor = None, sample_from: str = None):
        if sample_from:
            scores = self.process_filter(cond_scores=auxiliary_scores, guided_scores=scores, sample_from=sample_from)
        else:
            scores = self.process(scores)

        for t in removed_tokens or []:
            scores[:, t] = float("-inf")

        if self.do_sample:
            probs = scores.softmax(dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        return torch.argmax(scores, dim=-1)

@dataclass
class MiMoAudioOutput(ModelOutput):
    text_logits: torch.FloatTensor | None = None
    local_hidden_states: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    output_attentions: tuple | None = None
    """Downcast hidden states for local transformer generation"""


@dataclass
class MiMoAudioConfig(Qwen2Config):
    def __init__(
        self,
        *,
        speech_vocab_size: str | int = "1025-1025-129-129-129-129-129-129",
        speech_zeroemb_idx: str | int = "1024-1024-128-128-128-128-128-128",
        delay_pattern: str = "0-1-2-3-4-5-6-7",
        head_dim: int = 128,
        group_size: int = 4,
        audio_channels: int = 8,
        local_dim: int = 1024,
        local_layers: int = 16,
        local_attn_heads: int = 64,
        local_ffn_dim: int = 4096,
        local_attn_dropout: float = 0.1,
        input_local_layers: int = 6,
        input_local_dim: int | None = None,
        input_full_attention: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.speech_vocab_size = speech_vocab_size
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.delay_pattern = delay_pattern

        self.head_dim = head_dim

        self.group_size = group_size
        self.audio_channels = audio_channels

        self.local_dim = local_dim
        self.local_layers = local_layers
        self.local_attn_heads = local_attn_heads
        self.local_ffn_dim = local_ffn_dim
        self.local_attn_dropout = local_attn_dropout

        self.input_local_layers = input_local_layers
        self.input_local_dim = input_local_dim or local_dim

        self.input_full_attention = input_full_attention

    def _parse_maybe_list(self, value: str | int, length: int) -> List[int]:
        if isinstance(value, str) and "-" in value:
            return [int(s) for s in value.split("-")]
        return [int(value)] * length

    def parsed_speech_empty_ids(self):
        return self._parse_maybe_list(self.speech_zeroemb_idx, self.audio_channels)

    def parsed_speech_vocab_sizes(self):
        return self._parse_maybe_list(self.speech_vocab_size, self.audio_channels)

    def parsed_delay_pattern(self):
        return self._parse_maybe_list(self.delay_pattern, self.audio_channels)

    def local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.local_dim
        config.num_hidden_layers = self.local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = self.local_ffn_dim
        config.attention_dropout = self.local_attn_dropout

        return config

    def input_local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.input_local_dim
        config.num_hidden_layers = self.input_local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = config.hidden_size * 4
        config.attention_dropout = self.local_attn_dropout

        return config


@dataclass
class MiMoAudioArguments:
    model_name_or_path: str
    sosp_idx: int
    eosp_idx: int
    sostm_idx: int
    eostm_idx: int
    eot_idx: int
    empty_idx: int

    def to_dict(self):
        return {
            "model_name_or_path": self.model_name_or_path,
            "sosp_idx": self.sosp_idx,
            "eosp_idx": self.eosp_idx,
            "sostm_idx": self.sostm_idx,
            "eostm_idx": self.eostm_idx,
            "eot_idx": self.eot_idx,
            "empty_idx": self.empty_idx,
        }


def custom_qwen2_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1].cpu(),)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()

class MiMoAudioForCausalLM(Qwen2PreTrainedModel):
    def __init__(
        self,
        config: MiMoAudioConfig | Qwen2Config,
        args: MiMoAudioArguments | dict,
    ):
        super().__init__(config)
        config = (
            MiMoAudioConfig(**vars(config))
            if isinstance(config, Qwen2Config)
            else config
        )
        args = MiMoAudioArguments(**args) if isinstance(args, dict) else args
        self.config = config
        self.args = args

        self.model = Qwen2Model(config)
        self.model.forward = types.MethodType(custom_qwen2_forward, self.model)

        self.speech_vocab_sizes = config.parsed_speech_vocab_sizes()
        self.speech_empty_ids = config.parsed_speech_empty_ids()
        self.delay_pattern = config.parsed_delay_pattern()

        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Construct local transformer
        self.local_config = config.local_config()
        self.local_transformer = Qwen2Model(self.local_config)
        self.local_transformer.embed_tokens = None

        # Add input local transformer if configured
        self.input_local_config = config.input_local_config()
        self.input_local_transformer = Qwen2Model(self.input_local_config)
        self.input_local_transformer.embed_tokens = None

        self.local_transformer_lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    self.local_config.hidden_size,
                    self.speech_vocab_sizes[i],
                    bias=False,
                )
                for i in range(self.audio_channels)
            ]
        )

        self.speech_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    self.speech_vocab_sizes[i],
                    self.input_local_config.hidden_size,
                    padding_idx=self.speech_empty_ids[i],
                )
                for i in range(self.audio_channels)
            ]
        )

        if self.input_local_config.hidden_size != self.local_config.hidden_size:
            self.speech_embeddings_to_local = nn.Linear(
                self.input_local_config.hidden_size,
                self.local_config.hidden_size,
                bias=False,
            )
        else:
            self.speech_embeddings_to_local = None

        # Create speech_group_downcast_first for group_first_in_global_context
        self.speech_group_downcast = nn.Linear(
            self.input_local_config.hidden_size * config.group_size,
            config.hidden_size,
            bias=False,
        )

        self.hidden_states_downcast = nn.Linear(
            config.hidden_size,
            self.local_config.hidden_size,
            bias=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor):
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Process each group independently: [B*T//group_size, group_size, hidden_size]
        input_embeddings = speech_embeddings.reshape(
            B * T_groups, group_size, hidden_size
        )

        output: BaseModelOutputWithPast = self.input_local_transformer(
            inputs_embeds=input_embeddings,
            return_dict=True,
            is_causal=not self.config.input_full_attention,  # for SDPA
        )
        encoded_embeddings = output.last_hidden_state

        # Reshape back to original format
        # [B*T//group_size, group_size, hidden_size] -> [B, T//group_size, group_size, hidden_size]
        encoded_embeddings = encoded_embeddings.reshape(
            B, T_groups, group_size, hidden_size
        )

        return encoded_embeddings

    def _prepare_input_embeds(
        self,
        input_ids: torch.LongTensor,  # [B, audio_channels + 1, new_T]
    ):
        B = input_ids.shape[0]

        input_ids = input_ids.int()
        group_size = self.config.group_size

        text_input_ids = input_ids[:, 0, ::group_size]
        speech_input_ids = (
            input_ids[:, 1:, :]
            .view(B, self.audio_channels, -1, group_size)
            .transpose(1, 2)
        )  # [B, T//group_size, audio_channels, group_size]

        is_speech = text_input_ids == self.args.empty_idx  # [B, T//group_size]

        speech_embeds = torch.zeros(
            (
                B,
                is_speech.shape[1],
                group_size,
                self.input_local_config.hidden_size,
            ),
            device=input_ids.device,
            dtype=torch.bfloat16,
        )

        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]
            cur_speech_embeds: torch.Tensor = cur_embed(cur_speech_ids)
            # [B, T_groups, group_size, hidden_size]

            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)

            speech_embeds += cur_speech_embeds

        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        # Apply input local transformer if configured
        speech_embeds = self.apply_input_local_transformer(speech_embeds)
        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        T_groups = speech_embeds.shape[1]
        speech_grouped_embeds: torch.Tensor = self.speech_group_downcast(
            speech_embeds.view(B, T_groups, -1)
        )  # [B, T_groups, hidden_size]

        text_embeds: torch.Tensor = self.model.embed_tokens(text_input_ids)
        text_zero_mask = text_input_ids == self.args.empty_idx
        text_embeds.masked_fill_(text_zero_mask.unsqueeze(-1), 0.0)

        return text_embeds + speech_grouped_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, audio_channels + 1, new_T]
        attention_mask: torch.Tensor,  # [B, T_group]
        position_ids: torch.LongTensor,  # [B, new_T_group]
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,  # [new_T_group]
        **_kwargs,
    ):
        inputs_embeds = self._prepare_input_embeds(input_ids)

        output_attentions = _kwargs.get("output_attentions", None)

        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = outputs.last_hidden_state  # [B, new_T_group, hidden_size]

        text_logits: torch.Tensor = self.lm_head(
            hidden_states[:, -1:, :]
        )  # [B, 1, vocab_size]
        shift_hidden_states: torch.Tensor = self.hidden_states_downcast(
            hidden_states[:, -1:, :]
        )  # [B, 1, hidden_size]

        return MiMoAudioOutput(
            text_logits=text_logits,
            local_hidden_states=shift_hidden_states,
            past_key_values=outputs.past_key_values,
            output_attentions=outputs.attentions
        )

    def local_forward(
        self,
        local_embeds: torch.FloatTensor,  # [B, 1, hidden_size]
        tokens_dtype: torch.dtype,
        tokens_device: torch.device,
        local_sampler: MiMoSampler | None = None,
    ):
        B = local_embeds.shape[0]
        delay_iters = self.group_size + max(self.delay_pattern)
        past_key_values = DynamicCache()
        local_tokens = torch.zeros(
            (B, self.group_size, self.audio_channels),
            dtype=tokens_dtype,
            device=tokens_device,
        )
        if local_sampler is None:
            local_sampler = MiMoSampler()

        for t in range(delay_iters):
            output: BaseModelOutputWithPast = self.local_transformer(
                inputs_embeds=local_embeds,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            hidden_state = output.last_hidden_state
            past_key_values = output.past_key_values

            local_embeds = torch.zeros_like(local_embeds)
            for idx in range(self.audio_channels):
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size
                cur_empty = self.speech_empty_ids[idx]
                if cur_start <= t < cur_end:
                    cur_lm_head = self.local_transformer_lm_heads[idx]
                    cur_scores: torch.Tensor = cur_lm_head(hidden_state)[:, -1, :]
                    # [B, vocab_size]
                    cur_token = local_sampler.sample(
                        cur_scores,
                        [cur_empty],
                    )

                    local_tokens[:, t - cur_start, idx] = cur_token
                    cur_input_embed = self.speech_embeddings[idx](
                        cur_token.unsqueeze(1)
                    )
                    if self.speech_embeddings_to_local is not None:
                        cur_input_embed = self.speech_embeddings_to_local(
                            cur_input_embed
                        )
                    local_embeds += cur_input_embed

        return local_tokens  # [B, group_size, audio_channels]

    def _prepare_attention_mask(
        self, inputs: torch.Tensor, input_ids_length: int
    ) -> torch.Tensor:
        # No information for attention mask inference -> return default attention mask
        return torch.ones(
            (inputs.shape[0], input_ids_length),
            dtype=torch.bool,
            device=inputs.device,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        input_ids = input_ids.reshape(
            input_ids.shape[0], -1, (self.audio_channels + 1) * self.config.group_size
        ).transpose(1, 2)  # [B, audio_channels*group_size, T]
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = (
                past_key_values[0][0].shape[2] if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_length,
                input_ids.shape[2],
                dtype=torch.long,
                device=input_ids.device,
            )

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if (
                inputs_embeds is not None or cache_position[-1] >= input_ids.shape[2]
            ):  # Exception 1 or Exception 3
                input_ids = input_ids[:, :, -cache_position.shape[0] :]
            elif (
                input_ids.shape[2] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, :, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = (
            "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        )
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(
                    memory_format=torch.contiguous_format
                )
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(
                memory_format=torch.contiguous_format
            )

        # 4. Create missing `position_ids` on the fly
        if attention_mask is not None and kwargs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = (
                position_ids  # placed in kwargs for further processing (see below)
            )

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input: torch.Tensor = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[2] :]
                    model_input = model_input.clone(
                        memory_format=torch.contiguous_format
                    )
                model_inputs[model_input_name] = model_input

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        if model_inputs[input_ids_key] is not None:
            model_inputs[input_ids_key] = (
                cast(torch.Tensor, model_inputs[input_ids_key])
                .transpose(1, 2)
                .reshape(input_ids.shape[0], -1, (self.audio_channels + 1))
                .transpose(1, 2)
            )  # [B, audio_channels, T*group_size]

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

    def _get_initial_cache_position(self, input_ids: torch.Tensor, model_kwargs: dict):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            cache_position = (
                torch.ones_like(
                    model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64
                ).cumsum(0)
                - 1
            )
        else:
            cache_position = (
                torch.ones(
                    (
                        input_ids.shape[1]
                        // (self.audio_channels + 1)
                        // self.config.group_size,
                    ),
                    dtype=torch.int64,
                    device=input_ids.device,
                ).cumsum(0)
                - 1
            )

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif (
                hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None
            ):
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position

        return model_kwargs

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        stopping_criteria: StoppingCriteriaList | list | None = None,
        streamer: BaseStreamer | None = None,
        synced_gpus: bool | None = None,
        global_sampler: MiMoSampler | None = None,
        local_sampler: MiMoSampler | None = None,
        warmup_run: bool | None = None,
        turn_indices: list[dict] | None = None,
        guidance_scale: float = 1.0,
        sample_from: str | None = None,
        tokenizer = None,
        intra: str | None = "sum",
        inter: float | None = 1.0,
        layer_from: str | None = "last_1",
        num_rounds: int | None = 2,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )

        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 3. Define model inputs
        input_ids, _model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        input_ids_length = input_ids.shape[-1]
        input_ids_length //= self.group_size * (self.audio_channels + 1)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        if "attention_mask" not in model_kwargs:
            model_kwargs["attention_mask"] = self._prepare_attention_mask(
                inputs, input_ids_length
            )

        device = input_ids.device
        self._prepare_special_tokens(generation_config, True, device=device)

        model_kwargs["use_cache"] = True
        model_kwargs["past_key_values"] = DynamicCache()

        prepared_stopping_criteria = StoppingCriteriaList(
            stopping_criteria if stopping_criteria is not None else []
        )
        prepared_stopping_criteria.append(
            MiMoStopper(
                self.group_size,
                self.audio_channels,
                max_length=generation_config.max_length,
            )
        )
        stance = "default" if warmup_run else "eager_on_recompile"
        with torch.compiler.set_stance(stance):
            return self.slm_sample(
                input_ids,
                turn_indices=turn_indices,
                sample_from=sample_from,
                guidance_scale=guidance_scale,
                intra=intra,
                inter=inter,
                layer_from=layer_from,
                num_rounds=num_rounds,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                global_sampler=global_sampler,
                local_sampler=local_sampler,
                tokenizer=tokenizer,
                **model_kwargs,
            )

    def slm_sample(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: BaseStreamer | None,
        global_sampler: MiMoSampler | None = None,
        local_sampler: MiMoSampler | None = None,
        guidance_scale: float = 1.0,
        turn_indices: List[dict] | None = None,
        sample_from: str | None = None,
        intra: str | None = "sum",
        inter: float | None = 1.0,
        layer_from: str | None = "last_1",
        num_rounds: int | None = 2,
        tokenizer = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        max_length = generation_config.max_length

        # Audio-adapted Context-Aware Decoding (paper Sec. 2.2).
        B_orig = input_ids.shape[0]
        use_cfg = guidance_scale > 1.0 and turn_indices

        if use_cfg:
            with torch.no_grad():
                # Probe pass: a single forward to harvest attention maps for the
                # latent-awareness scoring (paper Sec. 2.3). Cache is disabled
                # because nothing past this pass needs to be reused.
                probe_kwargs = model_kwargs.copy()
                if "past_key_values" in probe_kwargs:
                    del probe_kwargs["past_key_values"]
                probe_kwargs["use_cache"] = False

                probe_inputs = self.prepare_inputs_for_generation(input_ids, **probe_kwargs)
                probe_inputs['output_attentions'] = True
                outputs_probe = self(**probe_inputs)

                # Layer selection: last_1 / last_4 / all (paper Sec. 2.3 (1)).
                attentions = outputs_probe.output_attentions  # tuple of (B, heads, S, S)
                if layer_from == "last_1":
                    selected_attns = attentions[-1:]
                elif layer_from == "last_4":
                    selected_attns = attentions[-4:]
                elif layer_from == "all":
                    selected_attns = attentions
                else:
                    selected_attns = attentions[-1:]

                # Average over selected layers and heads, then take the last
                # query token's row to get per-key-position attention -> (B, S).
                stacked_attns = torch.stack(selected_attns, dim=0)
                avg_attn = stacked_attns.mean(dim=(0, 2))
                last_token_attn = avg_attn[:, -1, :]

                # Round-level scoring + top-K masking (paper Sec. 2.3 (2)-(4)).
                # `stride` maps turn_indices (in flat-token units) into the
                # MiMo grouped-token sequence used by the attention map.
                high_attn_mask = torch.zeros_like(last_token_attn, dtype=torch.bool)
                stride = getattr(self, 'group_size', 1)

                def get_intra_score(attn_slice, method):
                    if attn_slice.numel() == 0:
                        return 0.0
                    if method == "sum":
                        return attn_slice.sum().item()
                    if method == "mean":
                        return attn_slice.mean().item()
                    if method == "max":
                        return attn_slice.max().item()
                    return 0.0

                for b in range(B_orig):
                    b_attn = last_token_attn[b]
                    rounds_info = []
                    num_turns = len(turn_indices)

                    # Index of the current (unanswered) user turn; the round
                    # containing it is excluded from key-context scoring.
                    last_user_idx = num_turns - 2 if num_turns % 2 == 0 else num_turns - 1

                    for i in range(0, last_user_idx, 2):
                        u_start = turn_indices[i]['start_idx'] // stride
                        u_end = turn_indices[i + 1]['start_idx'] // stride
                        a_start = turn_indices[i + 1]['start_idx'] // stride
                        a_end = turn_indices[i + 2]['start_idx'] // stride

                        u_score = get_intra_score(b_attn[u_start:u_end], intra)
                        a_score = get_intra_score(b_attn[a_start:a_end], intra)

                        # Round score = S(A) + inter * S(U)  (paper Eq. 4).
                        round_score = (u_score * inter) + a_score
                        rounds_info.append({
                            'round_idx': i // 2,
                            'u_start': u_start,
                            'a_end': a_end,
                            'score': round_score,
                        })

                    sorted_rounds = sorted(rounds_info, key=lambda x: x['score'], reverse=True)
                    top_rounds = sorted_rounds[:num_rounds]

                    # Mark the top-K rounds as the key context Ckey; these
                    # positions will be zeroed in the unconditional pass.
                    for r in top_rounds:
                        high_attn_mask[b, r['u_start']: r['a_end']] = True

                # Safety: never mask the current user turn or the generation
                # tokens — the model must always be able to attend to them.
                if num_turns > 0:
                    last_user_start = turn_indices[last_user_idx]['start_idx'] // stride
                    high_attn_mask[:, last_user_start:] = False
                high_attn_mask[:, -1] = False

                # Construct the unconditional attention mask by zeroing the
                # key-context positions in the original mask.
                base_mask = probe_inputs['attention_mask']
                uncond_mask = base_mask.clone()
                uncond_mask[high_attn_mask] = 0

                del outputs_probe, probe_inputs

            input_ids = torch.cat([input_ids, input_ids], dim=0)

            if "attention_mask" in model_kwargs:
                cond_mask = model_kwargs["attention_mask"]
                combined_mask = torch.cat([cond_mask, uncond_mask], dim=0)
                model_kwargs["attention_mask"] = combined_mask

        B, cur_len = input_ids.shape
        cur_len //= self.group_size * (self.audio_channels + 1)
        initial_len = cur_len
        this_peer_finished = False
        unfinished_sequences = torch.ones(B, dtype=torch.long, device=input_ids.device)
        
        min_length = 0
        stop_token_ids = set()
        for criterion in stopping_criteria:
            if isinstance(criterion, MiMoStopper):
                if criterion.min_length is not None:
                    min_length = max(min_length, criterion.min_length)
                stop_token_ids.update(criterion.stop_token_ids)

        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            if (
                cast(torch.Tensor, model_inputs["input_ids"]).shape[2]
                != self.group_size
            ):
                # prefill run
                with torch.compiler.set_stance("force_eager"):
                    outputs: MiMoAudioOutput = self(**model_inputs)
            else:
                outputs: MiMoAudioOutput = self(**model_inputs)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Text CFG
            text_logits: torch.Tensor = outputs.text_logits[:, -1, :].clone() # [B or 2B, vocab_size]

            if use_cfg:
                # CAD logit combination (paper Eq. 3):
                #   l_tilde = log P(uncond) + alpha * (log P(cond) - log P(uncond))
                cond_logits, uncond_logits = text_logits.chunk(2)
                text_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

            removed_tokens = None
            if cur_len < min_length:
                removed_tokens = list(stop_token_ids)

            if use_cfg:
                next_text_tokens = global_sampler.sample(text_logits, removed_tokens=removed_tokens, auxiliary_scores=cond_logits, sample_from=sample_from)
            else:
                next_text_tokens = global_sampler.sample(text_logits, removed_tokens=removed_tokens) # [B]

            local_hidden_states = outputs.local_hidden_states

            # The local forward only consumes the conditional (first B_orig)
            # half of the dual-stream batch.
            if use_cfg:
                local_hidden_states = local_hidden_states[:B_orig]

            # Only Supports batch_size=1 here
            if next_text_tokens[0] != self.args.empty_idx:
                zero_embed_tensor = torch.tensor(
                    self.speech_empty_ids,
                    device=next_text_tokens.device,
                    dtype=input_ids.dtype,
                )
                next_speech_tokens = zero_embed_tensor.view(
                    1, 1, self.audio_channels
                ).expand(B_orig, self.config.group_size, -1)
            else:
                next_speech_tokens = self.local_forward(
                    local_embeds=local_hidden_states,
                    tokens_dtype=next_text_tokens.dtype,
                    tokens_device=next_text_tokens.device,
                    local_sampler=local_sampler,
                )

            next_text_tokens = next_text_tokens.reshape(B_orig, 1, 1).expand(
                -1, self.group_size, -1
            )  # [B, group_size, 1]

            # generate speech tokens
            next_tokens = torch.cat(
                (next_text_tokens, next_speech_tokens), dim=-1
            ).reshape(B_orig, -1)  # [B, group_size * (audio_channels + 1)]

            if use_cfg:
                next_tokens = torch.cat([next_tokens, next_tokens], dim=0)

            input_ids = torch.cat(
                [input_ids, next_tokens], dim=-1
            )  # [B or 2B, T*group_size*vq]

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, None
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        input_ids = input_ids[:B_orig]

        return input_ids

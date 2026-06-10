import os

import tqdm
import torch
from loguru import logger
from huggingface_hub import cached_assets_path

from kimia_infer.models.detokenizer import get_audio_detokenizer
from .prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler
from huggingface_hub import snapshot_download

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
import math, types, gc

from src.modeling_moonshot_kimia import MoonshotKimiaForCausalLM

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        padding_mask = None,
):
    # LlamaFlashAttention2 attention does not support output_attentions

    # output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dime x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos = cos[position_ids]
    sin = sin[position_ids]
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # TODO: llama does not have dropout in the config??
    # It is recommended to use dropout with FA according to the docs
    # when training.
    dropout_rate = 0.0  # if not self.training else self.attn_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        logger.warning_once(
            "The input hidden states seems to be silently casted in float32, this might be related to"
            " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            " float16."
        )

        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

    if output_attentions:
        # Fall back to an eager attention path so we can return attention
        # weights for the CAD probe pass. Upstream Kimi-Audio normally uses
        # FlashAttention, which discards these weights.
        query_states_mac = query_states.transpose(1, 2)
        key_states_mac = key_states.transpose(1, 2)
        value_states_mac = value_states.transpose(1, 2)

        # GQA: replicate KV heads to match the number of query heads.
        num_key_value_groups = self.num_heads // self.num_key_value_heads
        key_states_mac = repeat_kv(key_states_mac, num_key_value_groups)
        value_states_mac = repeat_kv(value_states_mac, num_key_value_groups)

        attn_weights = torch.matmul(query_states_mac, key_states_mac.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply the padding mask (1 = keep, 0 = mask) before causal masking.
        if padding_mask is not None:
            mask_condition = (padding_mask == 0)[:, None, None, :]
            attn_weights = attn_weights.masked_fill(mask_condition, float("-inf"))

        # Re-create the causal mask we would normally rely on FlashAttention's
        # causal=True flag for.
        if q_len > 1:
            causal_mask = torch.full(
                (q_len, kv_seq_len),
                float("-inf"),
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            causal_mask = torch.triu(causal_mask, diagonal=kv_seq_len - q_len + 1)
            causal_mask = causal_mask[None, None, :, :]
            attn_weights = attn_weights + causal_mask

        # fp32 softmax for numerical stability.
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_mac)

        # Match FlashAttention's output layout (bsz, q_len, num_heads, head_dim).
        attn_output = attn_output.transpose(1, 2)
        attn_weights = attn_weights.cpu()

    else:
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            padding_mask,
            q_len,
            dropout=dropout_rate,
        )

    if input_dtype == torch.float32:
        attn_output = attn_output.to(torch.float32)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

class KimiAudio(object):
    def __init__(self, model_path: str, load_detokenizer: bool = True):
        logger.info(f"Loading kimi-audio main model")

        if os.path.exists(model_path):
            # local path
            cache_path = model_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_path)

        logger.info(f"Looking for resources in {cache_path}")
        logger.info(f"Loading whisper model")
        self.alm = MoonshotKimiaForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map='auto',
        )

        for i in range(len(self.alm.model.layers)):
            self.alm.model.layers[i].self_attn.forward = types.MethodType(
                custom_forward, self.alm.model.layers[i].self_attn
            )

        # self.alm = self.alm.to(torch.cuda.current_device())

        model_config = self.alm.config
        self.kimia_text_audiodelaytokens = model_config.kimia_mimo_audiodelaytokens
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, kimia_token_offset=self.kimia_token_offset, kimia_text_audiodelaytokens=self.kimia_text_audiodelaytokens
        )

        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            # need to compile extension moudules for the first time, it may take several minutes.
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            # in this case, you're not allowed to generate audio(wav)
            self.detokenizer = None

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]

    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
        turn_indices = None,
        guidance_scale: float = 1.0,
        intra: str = "sum",
        inter: float = 1.0,
        layer_from: str = "last_1",
        num_rounds: int = 2,
    ):

        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        text_previous_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0

        # Audio-adapted Context-Aware Decoding (paper Sec. 2).
        # Stage 1: probe-pass attention -> top-K round masking -> dual-stream batch.
        B_orig = decoder_input_audio_ids.shape[0]
        use_cfg = guidance_scale > 1.0 and turn_indices is not None

        if use_cfg:
            with torch.no_grad():
                # Single forward pass purely to harvest attention maps.
                outputs_probe = self.alm.forward(
                    input_ids=decoder_input_audio_ids,
                    text_input_ids=decoder_input_text_ids,
                    whisper_input_feature=decoder_input_whisper_feature,
                    is_continuous_mask=decoder_is_continuous_mask,
                    position_ids=decoder_position_ids,
                    past_key_values=None,
                    return_dict=True,
                    use_cache=False,
                    output_attentions=True,
                )

                attentions = outputs_probe.attentions

                # Layer selection (paper Sec. 2.3 (1)).
                if layer_from == "last_1":
                    selected_attns = attentions[-1:]
                elif layer_from == "last_4":
                    selected_attns = attentions[-4:]
                elif layer_from == "all":
                    selected_attns = attentions
                else:
                    selected_attns = attentions[-1:]

                # Average over selected layers + heads, take the last query
                # token's attention over the past keys -> (B, S).
                stacked_attns = torch.stack(selected_attns, dim=0)
                avg_attn = stacked_attns.mean(dim=(0, 2))
                last_token_attn = avg_attn[:, -1, :]
                high_attn_mask = torch.zeros_like(last_token_attn, dtype=torch.bool)

                # Kimi-Audio uses a flat (non-grouped) token sequence.
                stride = 1

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
                    # Exclude the round containing the current (unanswered) user turn.
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

                    # Mark top-K rounds as Ckey; their positions will be
                    # zeroed in the unconditional pass.
                    for r in top_rounds:
                        high_attn_mask[b, r['u_start']: r['a_end']] = True

                # Safety: never mask the current user turn or generation tokens.
                if num_turns > 0:
                    last_user_start = turn_indices[last_user_idx]['start_idx'] // stride
                    high_attn_mask[:, last_user_start:] = False
                high_attn_mask[:, -1] = False

                base_mask = torch.ones_like(decoder_input_audio_ids, dtype=torch.bool)
                uncond_mask = base_mask.clone()
                high_attn_mask = high_attn_mask.to(uncond_mask.device)
                uncond_mask[high_attn_mask] = 0

                del outputs_probe
                del attentions
                del selected_attns
                del stacked_attns
                del avg_attn
                del last_token_attn
                del b_attn

                gc.collect()
                torch.cuda.empty_cache()

            # Stage 2: build the dual-stream batch (conditional first, unconditional second).
            decoder_input_audio_ids = torch.cat([decoder_input_audio_ids, decoder_input_audio_ids], dim=0)
            decoder_input_text_ids = torch.cat([decoder_input_text_ids, decoder_input_text_ids], dim=0)
            attention_mask = torch.cat([base_mask, uncond_mask], dim=0)

            if decoder_is_continuous_mask is not None:
                decoder_is_continuous_mask = torch.cat([decoder_is_continuous_mask, decoder_is_continuous_mask], dim=0)

            if decoder_input_whisper_feature is not None:
                decoder_input_whisper_feature = decoder_input_whisper_feature + decoder_input_whisper_feature

            # decoder_position_ids is intentionally kept at B_orig (sequence indices).

        else:
            attention_mask = None

        for i in tqdm.tqdm(
            range(max_new_tokens), desc="Generating tokens", disable=False
        ):
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
                attention_mask=attention_mask,
            )

            if use_cfg:
                # Audio Logits CFG
                a_cond, a_uncond = audio_logits.chunk(2)
                cfg_audio_logits = a_uncond + guidance_scale * (a_cond - a_uncond)
                audio_logits = torch.cat([a_cond[:, :-1, :], cfg_audio_logits[:, -1:, :]], dim=1)

                # Text Logits CFG
                t_cond, t_uncond = text_logits.chunk(2)
                cfg_text_logits = t_uncond + guidance_scale * (t_cond - t_uncond)
                text_logits = torch.cat([t_cond[:, :-1, :], cfg_text_logits[:, -1:, :]], dim=1)

            # Sample text token using the sampler
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )

            # Sample audio token using the sampler
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )

            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text

            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1

            previous_audio_tokens[i : i + 1] = next_audio_token

            audio_stream_is_finished = next_audio_token.item() in self.eod_ids

            if (
                output_type == "text"
                and text_stream_is_finished
                or output_type == "both"
                and audio_stream_is_finished
            ):
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens
            else:
                if use_cfg:
                    # Duplicate the next-step token along the batch dim so the
                    # dual-stream (cond + uncond) batch shape is preserved.
                    decoder_input_audio_ids = torch.cat([next_audio_token, next_audio_token], dim=0).unsqueeze(1)
                    decoder_input_text_ids = torch.cat([next_token_text, next_token_text], dim=0).unsqueeze(1)

                    # Generated tokens must always be attendable -> extend mask with 1s.
                    if attention_mask is not None:
                        new_mask_step = torch.ones((B_orig * 2, 1), dtype=attention_mask.dtype,
                                                   device=attention_mask.device)
                        attention_mask = torch.cat([attention_mask, new_mask_step], dim=1)
                else:
                    decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                    decoder_input_text_ids = next_token_text.unsqueeze(1)

                    if attention_mask is not None:
                        new_mask_step = torch.ones((B_orig, 1), dtype=attention_mask.dtype,
                                                   device=attention_mask.device)
                        attention_mask = torch.cat([attention_mask, new_mask_step], dim=1)

                decoder_position_ids = (
                    torch.zeros(1, 1, device=torch.cuda.current_device())
                    .fill_(last_position_id + 1)
                    .long()
                    .view(1, 1)
                )
                last_position_id += 1

                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None

        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
        guidance_scale = 1.0,
        intra = "sum",
        inter = 1.0,
        layer_from = "last_1",
        num_rounds = 2,
    ):
        ## TODO: 需要一个check函数，检查输入的history格式是否合法
        ## 比如，对于ASR任务，一定是: text-instruction/audio-instruction + audio-content, 我理解content和instruction是不能换位置的
        ## assistant前必须有user等等，我觉得最好做一下check

        assert output_type in ["text", "both"]

        history = self.prompt_manager.get_prompt(chats, output_type=output_type)

        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature

        if guidance_scale > 1.0:
            # Recover turn boundaries from the audio token stream by scanning
            # for Kimi-Audio's role-marker tokens. Kept on the audio stream
            # because that is where role markers live in this model.
            user_token_id = self.prompt_manager.extra_tokens.kimia_user_msg_start
            asst_token_id = self.prompt_manager.extra_tokens.kimia_assistant_msg_start

            user_indices = torch.where(audio_input_ids[0] == user_token_id)[0].tolist()
            asst_indices = torch.where(audio_input_ids[0] == asst_token_id)[0].tolist()

            extracted_turns = []
            for idx in user_indices:
                extracted_turns.append({'start_idx': idx, 'role': 'user'})
            for idx in asst_indices:
                extracted_turns.append({'start_idx': idx, 'role': 'assistant'})

            turn_indices = sorted(extracted_turns, key=lambda x: x['start_idx'])
        else:
            turn_indices = None

        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                max_new_tokens = 7500 - audio_input_ids.shape[1]

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]

        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
            turn_indices=turn_indices,
            guidance_scale=guidance_scale,
            intra=intra,
            inter=inter,
            layer_from=layer_from,
            num_rounds=num_rounds,
        )

        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]  #  filter out the illegal tokens

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset

        generated_text_tokens = [
            t for t in generated_text_tokens if t < self.kimia_token_offset
        ]
        generated_text = self.detokenize_text(generated_text_tokens)
        if self.detokenizer is not None and output_type == "both":
            generated_wav = self.detokenize_audio(generated_wav_tokens)
        else:
            generated_wav = None

        return generated_wav, generated_text

    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30  # hard-coded right now
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)

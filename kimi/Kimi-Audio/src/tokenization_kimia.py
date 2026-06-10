# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Union
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    Union,
    Optional,
)
from tiktoken.load import load_tiktoken_bpe
import tiktoken
from pathlib import Path
import os
import logging
from tokenizers import AddedToken

logger = logging.getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tiktoken.model"}


class TikTokenTokenizer(PreTrainedTokenizer):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 293 + 128

    pat_str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        bos_token: Union[str, AddedToken] = "[BOS]",
        eos_token: Union[str, AddedToken] = "[EOS]",
        unk_token: Union[str, AddedToken] = "[UNK]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        additional_special_tokens: Optional[List[str]] = None,
        added_tokens_decoder: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(vocab_file), vocab_file

        mergeable_ranks = load_tiktoken_bpe(vocab_file)
        num_base_tokens = len(mergeable_ranks)

        used_special_tokens = [
            "[BOS]",
            "[EOS]",
            "<|im_msg_end|>",  # 0
            "<|im_user_msg_start|>",  # 1
            "<|im_assistant_msg_start|>",  # 2
            "<|reserved_token_0|>",  # 3
            "<|reserved_token_1|>",
            "<|reserved_token_2|>",
            "<|reserved_token_3|>",  # 4
            "[EOT]",
            "<|reserved_token_4|>",  # 5
            "<|reserved_token_5|>",  # 6
            "<|reserved_token_6|>",  # 7
            "<|reserved_token_7|>",  # 8
            "<|reserved_token_8|>",  # 9
            "<|reserved_token_9|>",  # 10
            "<|reserved_token_10|>",  # 11
            "<|reserved_token_11|>",  # 12
            "<|im_media_begin|>",  # 13
            "<|reserved_token_12|>",  # 14
            "<|im_media_end|>",  # 15
            "<|reserved_token_13|>",  # 16
            "<|reserved_token_14|>",  # 17
            "<|im_kimia_text_blank|>",  # 18
            "<|im_kimia_text_eos|>",  # 19
            "<|reserved_token_15|>",  # 20
            "<|reserved_token_16|>",  # 21
            "<|im_kimia_user_msg_start|>",  # 22
            "<|im_kimia_assistant_msg_start|>",  # 23
            "<|reserved_token_17|>",  # 24
            "<|reserved_token_18|>",  # 25
            "<|reserved_token_19|>",  # 26
            "<|im_kimia_speech_ct_id|>",  # 27
            "<|im_kimia_speech_ctd_id|>",  # 28
        ]
        autoset_special_tokens = [
            f"<|reserved_token_{i}|>"
            for i in range(
                20, self.num_reserved_special_tokens - len(used_special_tokens) + 20
            )
        ]
        special_tokens = used_special_tokens + autoset_special_tokens
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(vocab_file).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {vocab_file}")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_token = "[BOS]"
        self.bos_id: int = self.special_tokens["[BOS]"]
        self.eos_token = "[EOS]"
        self.eos_id: int = self.special_tokens["[EOS]"]

        # use last speical token as pad token, the last - 1 is unk_token
        self.pad_token: str = special_tokens[-1]
        self.pad_id: int = self.special_tokens[self.pad_token]

        self.unk_token: str = special_tokens[-2]
        self.unk_id: int = self.special_tokens[self.pad_token]

        self.stop_tokens = {
            self.special_tokens["[EOS]"],
            self.special_tokens["[EOT]"],
        }

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    """ ----- Below are the abstract methods required by megatron ----- """

    @property
    def vocab_size(self):
        return self.n_words

    @property
    def vocab(self):
        if hasattr(self, "str_vocab"):
            return self.str_vocab
        self.str_vocab = {}

        # convert mergeable_ranks from bytes to string
        utf8_num, unicode_num = 0, 0
        for byte_key, index in self.model._mergeable_ranks.items():
            try:
                str_key = byte_key.decode("utf-8")
                utf8_num += 1
            except UnicodeDecodeError:
                # use backslashreplace so we can get num vocab different tokens
                # see: https://docs.python.org/3/howto/unicode.html
                # this vocab is only used for offline processing, so this is fine
                str_key = byte_key.decode("utf-8", "backslashreplace") + "_unicode_"
                unicode_num += 1

            self.str_vocab[str_key] = index
        logger.info(f"num utf8: {utf8_num}, num unicode: {unicode_num}")

        # add all special tokens to the dictionary
        self.str_vocab.update(self.model._special_tokens)

        assert len(self.str_vocab) == self.vocab_size
        return self.str_vocab

    @property
    def inv_vocab(self):
        return {v: k for k, v in self.vocab.items()}

    def tokenize(self, text, eos=True):
        # BOS: always add bos token
        # EOS:
        #    Most cases should be true when we are tokenizing a full sequence
        #    Only setting to false when we are running a inference
        return self.encode(text, bos=True, eos=eos)

    def detokenize(self, tokens):
        # convert tensor to list if needed...
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        return self.decode(tokens)

    @property
    def eod(self):
        return self.eos_id

    def bod(self):
        return self.bos_id

    @property
    def msk_start_id(self):
        return self.msk_start

    @property
    def msk_end_id(self):
        return self.msk_end

    def _get_index_2_bytes(self):
        if hasattr(self, "index_2_bytes"):
            return self.index_2_bytes

        # use array rather than dict for faster access
        self.index_2_bytes = [0] * self.model.n_vocab
        for byte_key, index in self.model._mergeable_ranks.items():
            self.index_2_bytes[index] = len(byte_key)

        for _, index in self.model._special_tokens.items():
            # in total we have 256 special tokens, 2^8 = 256
            # so the num of bytes of each token is only 1
            self.index_2_bytes[index] = 1

        return self.index_2_bytes

    def get_array_bytes(self, array):
        index_2_bytes = self._get_index_2_bytes()
        return sum(index_2_bytes[i] for i in array)

    @property
    def eos_token_id(self):
        return self.eos_id

    @property
    def pad_token_id(self):
        return self.pad_id

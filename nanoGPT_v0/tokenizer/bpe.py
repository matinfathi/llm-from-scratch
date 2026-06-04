"""
tokenizer/bpe.py - Tiktoken-backed BPE tokenizer.
"""

from __future__ import annotations

import tiktoken

from nanoGPT_v0.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Byte-pair encoding tokenizer using OpenAI's tiktoken encodings."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    # ── BaseTokenizer interface ────────────────────────────────────────────

    def fit(self, text: str) -> None:
        """Tiktoken encodings are pretrained, so there is nothing to fit."""

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._encoding.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    # ── Persistence ────────────────────────────────────────────────────────

    def _state(self) -> dict:
        return {"encoding_name": self.encoding_name}

    def _load_state(self, state: dict) -> None:
        self.encoding_name = state["encoding_name"]
        self._encoding = tiktoken.get_encoding(self.encoding_name)

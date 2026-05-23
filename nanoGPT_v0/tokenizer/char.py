"""
tokenizer/char.py — Character-level tokenizer (the original approach).
"""

from __future__ import annotations

from base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """Maps every unique character in the corpus to an integer id."""

    def __init__(self) -> None:
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    # ── BaseTokenizer interface ────────────────────────────────────────────

    def fit(self, text: str) -> None:
        chars = sorted(set(text))
        self._stoi = {c: i for i, c in enumerate(chars)}
        self._itos = {i: c for c, i in self._stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    # ── Persistence ────────────────────────────────────────────────────────

    def _state(self) -> dict:
        return {"stoi": self._stoi}

    def _load_state(self, state: dict) -> None:
        self._stoi = state["stoi"]
        self._itos = {int(i): c for c, i in self._stoi.items()}

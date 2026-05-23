"""
tokenizer/base.py — Abstract interface every tokenizer must implement.

To add a new tokenizer (e.g. BPE, SentencePiece):
  1. Subclass BaseTokenizer.
  2. Implement fit(), encode(), and decode().
  3. Register it in tokenizer/__init__.py.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):
    """Minimal tokenizer contract used throughout the project."""

    # ── Must implement ─────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, text: str) -> None:
        """Build vocabulary from raw text."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token ids."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token ids back to a string."""

    # ── Derived properties (subclasses may override) ───────────────────────

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""

    # ── Persistence helpers ────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise the tokenizer to a JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self._state(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BaseTokenizer":
        """Deserialise a tokenizer that was saved with .save()."""
        state = json.loads(Path(path).read_text())
        instance = cls.__new__(cls)
        instance._load_state(state)
        return instance

    # Subclasses may override these two to enable save/load:
    def _state(self) -> dict:
        raise NotImplementedError(f"{type(self).__name__} does not support serialisation.")

    def _load_state(self, state: dict) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support deserialisation.")

"""
tokenizer/__init__.py

Factory that returns the correct tokenizer from the TOKENIZER env key.
Adding a new tokenizer only requires:
  1. Subclassing BaseTokenizer in a new file.
  2. Registering it in the REGISTRY dict below.
"""

from __future__ import annotations

from base import BaseTokenizer
from char import CharTokenizer

REGISTRY: dict[str, type[BaseTokenizer]] = {
    "char": CharTokenizer,
    # "bpe": BPETokenizer,   ← add future tokenizers here
}


def build_tokenizer(name: str, text: str) -> BaseTokenizer:
    """Instantiate and fit a tokenizer by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown tokenizer '{name}'. Available: {list(REGISTRY)}")
    tokenizer = REGISTRY[name]()
    tokenizer.fit(text)
    return tokenizer


__all__ = ["BaseTokenizer", "CharTokenizer", "build_tokenizer"]

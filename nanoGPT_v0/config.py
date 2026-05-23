"""
config.py — Centralised, typed configuration loaded from .env
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _getf(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _geti(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


# ── Model configuration ───────────────────────────────────────────────────────


@dataclass
class GPTConfig:
    vocab_size: int = 65  # overwritten after tokenizer is built
    block_size: int = field(default_factory=lambda: _geti("BLOCK_SIZE", 256))
    n_layer: int = field(default_factory=lambda: _geti("N_LAYER", 6))
    n_head: int = field(default_factory=lambda: _geti("N_HEAD", 6))
    n_embed: int = field(default_factory=lambda: _geti("N_EMBED", 384))

    def __post_init__(self) -> None:
        if self.n_embed % self.n_head != 0:
            raise ValueError(f"n_embed ({self.n_embed}) must be divisible by n_head ({self.n_head})")


# ── Training configuration ────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # paths
    data_path: Path = field(default_factory=lambda: Path(_get("DATA_PATH", "data/shakespeare.txt")))
    tokenizer: str = field(default_factory=lambda: _get("TOKENIZER", "char"))
    checkpoint_dir: Path = field(default_factory=lambda: Path(_get("CHECKPOINT_DIR", "checkpoints")))
    log_file: Path = field(default_factory=lambda: Path(_get("LOG_FILE", "loss_log.json")))

    # optimiser
    max_steps: int = field(default_factory=lambda: _geti("MAX_STEPS", 5000))
    batch_size: int = field(default_factory=lambda: _geti("BATCH_SIZE", 64))
    max_lr: float = field(default_factory=lambda: _getf("MAX_LR", 1e-3))
    min_lr: float = field(default_factory=lambda: _getf("MIN_LR", 1e-4))
    warmup_steps: int = field(default_factory=lambda: _geti("WARMUP_STEPS", 100))
    grad_clip: float = field(default_factory=lambda: _getf("GRAD_CLIP", 1.0))
    weight_decay: float = field(default_factory=lambda: _getf("WEIGHT_DECAY", 0.01))
    train_split: float = field(default_factory=lambda: _getf("TRAIN_SPLIT", 0.9))

    # evaluation
    eval_interval: int = field(default_factory=lambda: _geti("EVAL_INTERVAL", 100))
    eval_iters: int = field(default_factory=lambda: _geti("EVAL_ITERS", 20))
    sample_interval: int = field(default_factory=lambda: _geti("SAMPLE_INTERVAL", 500))
    checkpoint_interval: int = field(default_factory=lambda: _geti("CHECKPOINT_INTERVAL", 1000))

    # generation / sampling
    sample_prompt: str = field(default_factory=lambda: _get("SAMPLE_PROMPT", "To be or not"))
    sample_max_new_tokens: int = field(default_factory=lambda: _geti("SAMPLE_MAX_NEW_TOKENS", 100))
    sample_temperature: float = field(default_factory=lambda: _getf("SAMPLE_TEMPERATURE", 0.8))
    sample_top_k: int = field(default_factory=lambda: _geti("SAMPLE_TOP_K", 40))

    def __post_init__(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

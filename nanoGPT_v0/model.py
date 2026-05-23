"""
model.py — GPT language model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from config import GPTConfig


# ── Building blocks ───────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head

        self.qkv_proj = nn.Linear(config.n_embed, config.n_embed * 3)
        self.out_proj = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x: Tensor) -> Tensor:
        B, T, E = x.shape
        q, k, v = self.qkv_proj(x).split(self.n_embed, dim=2)

        # (B, T, E) → (B, n_heads, T, head_dim)
        def reshape(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = nn.functional.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_proj(y)


class MLP(nn.Module):
    """Position-wise feed-forward network (4× expansion, tanh-GELU)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = config.n_embed * 4
        self.fc = nn.Linear(config.n_embed, hidden)
        self.gelu = nn.GELU(approximate="tanh")
        self.proj = nn.Linear(hidden, config.n_embed)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.gelu(self.fc(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual, LN → MLP → residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Top-level model ───────────────────────────────────────────────────────────


class GPT(nn.Module):
    """Decoder-only GPT language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                "wpe": nn.Embedding(config.block_size, config.n_embed),
                "h": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight tying: input embeddings ↔ output projection
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)

    # ── Initialisation ─────────────────────────────────────────────────────

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.transformer["wte"](idx) + self.transformer["wpe"](pos)

        for block in self.transformer["h"]:
            x = block(x)

        logits = self.lm_head(self.transformer["ln_f"](x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # ── Convenience ────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

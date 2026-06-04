"""
train.py — Training entry point for nanoGPT.

Usage:
    python train.py                   # uses .env defaults
    TOKENIZER=bpe python train.py     # override single env key inline
"""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from nanoGPT_v0.config import GPTConfig, TrainConfig
from nanoGPT_v0.model import GPT
from nanoGPT_v0.tokenizer import build_tokenizer, BaseTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Device ────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Data ──────────────────────────────────────────────────────────────────────


def load_dataset(
    cfg: TrainConfig,
    tokenizer: BaseTokenizer,
    device: torch.device,
) -> tuple[callable, callable]:
    """Return (get_train_batch, get_val_batch) closures."""
    text = cfg.data_path.read_text()
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    log.info("Dataset: %s tokens | vocab: %d", f"{len(tokens):,}", tokenizer.vocab_size)

    n = int(cfg.train_split * len(tokens))
    train_data, val_data = tokens[:n], tokens[n:]

    def _get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = int(cfg.data_path.parent.parent / ".env" and 0) or GPTConfig().block_size  # read lazily via config
        ix = torch.randint(len(data) - block_size - 1, (cfg.batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
        return x, y

    # Capture block_size once to avoid re-reading config in the hot loop
    block_size = GPTConfig().block_size

    def _make_batch(data: torch.Tensor):
        ix = torch.randint(len(data) - block_size - 1, (cfg.batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
        return x, y

    return lambda: _make_batch(train_data), lambda: _make_batch(val_data)


# ── Learning-rate schedule ────────────────────────────────────────────────────


def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Generation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate(
    model: GPT,
    prompt: str,
    tokenizer: BaseTokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            top_values, _ = torch.topk(logits, top_k)
            logits[logits < top_values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return tokenizer.decode(idx[0].tolist())


# ── Checkpointing ─────────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    step: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: BaseTokenizer,
) -> None:
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config,
            "tokenizer_state": tokenizer._state(),
            "tokenizer_class": type(tokenizer).__name__,
        },
        path,
    )
    log.info("Checkpoint saved → %s", path)


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: GPT, get_val_batch: callable, eval_iters: int) -> float:
    model.eval()
    losses = [model(*get_val_batch())[1].item() for _ in range(eval_iters)]
    model.train()
    return sum(losses) / len(losses)


# ── Training loop ─────────────────────────────────────────────────────────────


def train() -> GPT:
    train_cfg = TrainConfig()
    device = get_device()
    log.info("Device: %s", device)

    # Build tokenizer
    text = train_cfg.data_path.read_text()
    tokenizer = build_tokenizer(train_cfg.tokenizer, text)
    tokenizer.save(train_cfg.checkpoint_dir / "tokenizer.json")
    log.info("Tokenizer: %s | vocab_size: %d", train_cfg.tokenizer, tokenizer.vocab_size)

    # Build model
    model_cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(model_cfg).to(device)
    log.info(
        "Model: %dL / %dH / %dD | %.1fM params",
        model_cfg.n_layer,
        model_cfg.n_head,
        model_cfg.n_embed,
        model.num_parameters() / 1e6,
    )

    # Data loaders
    get_train_batch, get_val_batch = load_dataset(train_cfg, tokenizer, device)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.max_lr,
        weight_decay=train_cfg.weight_decay,
    )

    # Logging
    loss_log: dict[str, list] = {"steps": [], "train": [], "val": []}

    pbar = tqdm(range(train_cfg.max_steps), desc="Training")
    for step in pbar:
        # ── Evaluate ──────────────────────────────────────────────────────
        if step % train_cfg.eval_interval == 0:
            val_loss = evaluate(model, get_val_batch, train_cfg.eval_iters)
            loss_log["val"].append(val_loss)
            tqdm.write(f"Step {step:5d} | val loss: {val_loss:.4f}")

        # ── Update LR ─────────────────────────────────────────────────────
        lr = cosine_lr_with_warmup(
            step,
            train_cfg.warmup_steps,
            train_cfg.max_steps,
            train_cfg.max_lr,
            train_cfg.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Forward / backward ────────────────────────────────────────────
        model.train()
        x, y = get_train_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        loss_log["steps"].append(step)
        loss_log["train"].append(loss.item())

        # ── Generate sample ───────────────────────────────────────────────
        if step > 0 and step % train_cfg.sample_interval == 0:
            sample = generate(
                model,
                train_cfg.sample_prompt,
                tokenizer,
                max_new_tokens=train_cfg.sample_max_new_tokens,
                temperature=train_cfg.sample_temperature,
                top_k=train_cfg.sample_top_k,
            )
            tqdm.write(f"\n--- Step {step} sample ---\n{sample}\n---\n")

        # ── Checkpoint ────────────────────────────────────────────────────
        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(
                train_cfg.checkpoint_dir / f"checkpoint_{step}.pt",
                step,
                model,
                optimizer,
                tokenizer,
            )

    # ── Final saves ───────────────────────────────────────────────────────────
    save_checkpoint(
        train_cfg.checkpoint_dir / "checkpoint_final.pt",
        train_cfg.max_steps,
        model,
        optimizer,
        tokenizer,
    )
    train_cfg.log_file.write_text(json.dumps(loss_log, indent=2))
    log.info("Loss log saved → %s", train_cfg.log_file)

    return model


if __name__ == "__main__":
    train()

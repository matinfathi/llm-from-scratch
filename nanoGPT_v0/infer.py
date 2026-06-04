"""
infer.py - Generate text from a trained nanoGPT checkpoint.

Usage:
    uv run python -m nanoGPT_v0.infer "To be or not"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanoGPT_v0.config import TrainConfig
from nanoGPT_v0.model import GPT
from nanoGPT_v0.tokenizer import BaseTokenizer, BPETokenizer, CharTokenizer


TOKENIZER_CLASSES: dict[str, type[BaseTokenizer]] = {
    "BPETokenizer": BPETokenizer,
    "CharTokenizer": CharTokenizer,
}


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer_from_state(class_name: str, state: dict) -> BaseTokenizer:
    if class_name not in TOKENIZER_CLASSES:
        raise ValueError(f"Unknown tokenizer class in checkpoint: {class_name}")

    tokenizer_cls = TOKENIZER_CLASSES[class_name]
    tokenizer = tokenizer_cls.__new__(tokenizer_cls)
    tokenizer._load_state(state)
    return tokenizer


def infer_tokenizer_class(state: dict) -> str:
    if "encoding_name" in state:
        return "BPETokenizer"
    if "stoi" in state:
        return "CharTokenizer"
    raise ValueError("Could not infer tokenizer class from tokenizer state.")


def load_tokenizer(checkpoint: dict, tokenizer_path: Path) -> BaseTokenizer:
    if "tokenizer_state" in checkpoint:
        state = checkpoint["tokenizer_state"]
        class_name = checkpoint.get("tokenizer_class") or infer_tokenizer_class(state)
        return load_tokenizer_from_state(class_name, state)

    state = json.loads(tokenizer_path.read_text())
    return load_tokenizer_from_state(infer_tokenizer_class(state), state)


@torch.no_grad()
def generate(
    model: GPT,
    prompt: str,
    tokenizer: BaseTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, k)
                logits[logits < top_values[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return tokenizer.decode(idx[0].tolist())


def parse_args() -> argparse.Namespace:
    train_cfg = TrainConfig()
    default_checkpoint = train_cfg.checkpoint_dir / "checkpoint_final.pt"
    default_tokenizer = train_cfg.checkpoint_dir / "tokenizer.json"

    parser = argparse.ArgumentParser(description="Generate text from a trained nanoGPT checkpoint.")
    parser.add_argument("prompt", help="Input sentence/prompt to continue.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Path to model checkpoint.")
    parser.add_argument("--tokenizer", type=Path, default=default_tokenizer, help="Path to tokenizer.json fallback.")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Use 0 for greedy.")
    parser.add_argument("--top-k", type=int, default=40, help="Keep only the top-k logits before sampling. Use 0 to disable.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, mps, etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = load_tokenizer(checkpoint, args.tokenizer)

    model = GPT(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output = generate(
        model,
        args.prompt,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build and overfit a tiny word-level transformer on 100 world stories.

This is intentionally one self-contained file. It implements:

* a word-level vocabulary;
* learned token and positional embeddings;
* padding attention masks;
* one- or two-layer transformer encoders with explicit Q, K, and V;
* a classifier for the dataset's five carrier answers; and
* training, evaluation, and resumable checkpointing.

Run from the repository root:

    python3 train_tiny_transformer.py

The default experiment selects 20 examples for each of the five answers and
trains until it reaches at least 98% accuracy on those same 100 examples.
This is an overfitting smoke test, not a generalization result.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>"]
LABELS = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
OBJECTS = ["Hairbrush", "Sneakers", "Glasses", "Key"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/week6_full_dataset/train.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--layers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tiny_transformer_overfit.pt"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="CPU is the portable, deterministic default.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenization while retaining punctuation."""

    return TOKEN_PATTERN.findall(text.lower())


def load_balanced_records(path: Path, total: int) -> list[dict[str, Any]]:
    """Select a deterministic subset balanced by queried object and answer."""

    cell_count = len(OBJECTS) * len(LABELS)
    if total <= 0 or total % cell_count != 0:
        raise ValueError(f"--examples must be a positive multiple of {cell_count}")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate it with "
            "python3 scripts/generate_week6_full_dataset.py"
        )

    quota = total // cell_count
    selected: dict[tuple[str, str], list[dict[str, Any]]] = {
        (object_name, label): [] for object_name in OBJECTS for label in LABELS
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "story_text" not in record or "answer" not in record:
                raise ValueError(f"{path}:{line_number} lacks story_text or answer")
            answer = record["answer"]
            object_name = record.get("question", {}).get("object")
            cell = (object_name, answer)
            if cell in selected and len(selected[cell]) < quota:
                selected[cell].append(record)
            if all(len(records) == quota for records in selected.values()):
                break

    missing = {cell: quota - len(records) for cell, records in selected.items()}
    missing = {cell: count for cell, count in missing.items() if count}
    if missing:
        raise ValueError(f"Dataset cannot supply a balanced subset: {missing}")

    # Interleave all 20 cells so a deterministic unshuffled view stays balanced.
    return [
        selected[(object_name, label)][i]
        for i in range(quota)
        for object_name in OBJECTS
        for label in LABELS
    ]


def build_vocabulary(texts: list[str]) -> tuple[dict[str, int], list[str]]:
    counts = Counter(token for text in texts for token in tokenize(text))
    index_to_token = SPECIAL_TOKENS + sorted(counts)
    token_to_index = {token: index for index, token in enumerate(index_to_token)}
    return token_to_index, index_to_token


def encode_records(
    records: list[dict[str, Any]], token_to_index: dict[str, int]
) -> tuple[Tensor, Tensor, int]:
    encoded = [
        [token_to_index["<cls>"]]
        + [token_to_index.get(token, token_to_index["<unk>"]) for token in tokenize(r["story_text"])]
        for r in records
    ]
    maximum_length = max(map(len, encoded))
    pad_index = token_to_index["<pad>"]
    inputs = torch.full((len(encoded), maximum_length), pad_index, dtype=torch.long)
    for row, token_ids in enumerate(encoded):
        inputs[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    targets = torch.tensor(
        [label_to_index[record["answer"]] for record in records], dtype=torch.long
    )
    return inputs, targets, maximum_length


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with explicit Q=XW_Q, K=XW_K, and V=XW_V."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_projection = nn.Linear(d_model, d_model, bias=False)
        self.k_projection = nn.Linear(d_model, d_model, bias=False)
        self.v_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        batch_size, sequence_length, d_model = x.shape

        # Before reshaping: Q, K, V are [B, T, D].
        # After reshaping: each attention head is [B, T, d_k].
        def split_heads(projected: Tensor) -> Tensor:
            return projected.view(
                batch_size, sequence_length, self.heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(self.q_projection(x))
        keys = split_heads(self.k_projection(x))
        values = split_heads(self.v_projection(x))

        # [B,H,T,d_k] @ [B,H,d_k,T] -> scores [B,H,T,T].
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)

        # True marks padding keys that no query is allowed to attend to.
        key_mask = padding_mask[:, None, None, :]
        scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))

        # [B,H,T,T] @ [B,H,T,d_k] -> [B,H,T,d_k].
        attended = weights @ values
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, d_model
        )
        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, heads: int, ff_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), padding_mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        maximum_length: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        pad_index: int,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index
        self.token_embedding = nn.Embedding(
            vocabulary_size, d_model, padding_idx=pad_index
        )
        self.position_embedding = nn.Embedding(maximum_length, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, ff_dim, dropout)
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, len(LABELS))

    def forward(self, token_ids: Tensor) -> Tensor:
        batch_size, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        padding_mask = token_ids.eq(self.pad_index)
        for block in self.blocks:
            x = block(x, padding_mask)
        # Token zero is <cls>; its final representation summarizes the story.
        return self.classifier(self.final_norm(x[:, 0]))


@torch.no_grad()
def accuracy(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    model.eval()
    predictions = model(inputs).argmax(dim=-1)
    return predictions.eq(targets).float().mean().item()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocabulary: list[str],
    configuration: dict[str, Any],
    training_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocabulary": vocabulary,
            "labels": LABELS,
            "configuration": configuration,
            "training_accuracy": training_accuracy,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(args.device)

    records = load_balanced_records(args.data, args.examples)
    texts = [record["story_text"] for record in records]
    token_to_index, vocabulary = build_vocabulary(texts)
    inputs, targets, maximum_length = encode_records(records, token_to_index)
    inputs, targets = inputs.to(device), targets.to(device)

    configuration = {
        "examples": args.examples,
        "layers": args.layers,
        "d_model": args.d_model,
        "heads": args.heads,
        "head_dim": args.d_model // args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "maximum_length": maximum_length,
        "seed": args.seed,
    }
    model = TinyTransformer(
        vocabulary_size=len(vocabulary),
        maximum_length=maximum_length,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pad_index=token_to_index["<pad>"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    loss_function = nn.CrossEntropyLoss()
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if checkpoint["vocabulary"] != vocabulary:
            raise ValueError("Checkpoint vocabulary does not match this dataset subset")
        if checkpoint["configuration"] != configuration:
            raise ValueError("Checkpoint model configuration does not match arguments")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resumed checkpoint at epoch {checkpoint['epoch']}")

    head_dim = args.d_model // args.heads
    print("Tiny-transformer 100-example overfitting test")
    print(f"Examples: {len(records)} ({args.examples // len(LABELS)} per label)")
    print(
        f"Coverage: {args.examples // (len(OBJECTS) * len(LABELS))} "
        "per queried-object/answer combination"
    )
    print(f"Vocabulary: {len(vocabulary)} word/punctuation tokens")
    print(f"Sequence tensor: [B,T] = [{args.batch_size},{maximum_length}]")
    print(f"Embedded X: [B,T,D] = [{args.batch_size},{maximum_length},{args.d_model}]")
    print(f"For one head, W_Q/W_K/W_V: [{args.d_model},{head_dim}]")
    print(f"For one head, Q/K/V: [{args.batch_size},{maximum_length},{head_dim}]")
    print(
        f"QK^T and attention mask: "
        f"[{args.batch_size},{maximum_length},{maximum_length}]"
    )
    print(f"One-head output: [{args.batch_size},{maximum_length},{head_dim}]")
    print(f"Classifier logits: [{args.batch_size},{len(LABELS)}] -> {LABELS}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    final_epoch = start_epoch - 1
    final_loss = float("nan")
    final_accuracy = accuracy(model, inputs, targets)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(inputs), device=device)
        total_loss = 0.0
        for offset in range(0, len(inputs), args.batch_size):
            indices = permutation[offset : offset + args.batch_size]
            logits = model(inputs[indices])
            loss = loss_function(logits, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(indices)

        final_epoch = epoch
        final_loss = total_loss / len(inputs)
        final_accuracy = accuracy(model, inputs, targets)
        if epoch == 1 or epoch % 10 == 0 or final_accuracy >= args.target_accuracy:
            print(
                f"epoch {epoch:3d} | loss {final_loss:.4f} "
                f"| memorization accuracy {final_accuracy:.1%}"
            )
        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                args.checkpoint,
                epoch,
                model,
                optimizer,
                vocabulary,
                configuration,
                final_accuracy,
            )
        if final_accuracy >= args.target_accuracy:
            break

    save_checkpoint(
        args.checkpoint,
        final_epoch,
        model,
        optimizer,
        vocabulary,
        configuration,
        final_accuracy,
    )
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    if final_accuracy < args.target_accuracy:
        raise RuntimeError(
            f"Overfit test failed: {final_accuracy:.1%} < {args.target_accuracy:.1%}. "
            "Try more epochs or a larger --d-model."
        )
    print(
        f"PASS: memorized {final_accuracy:.1%} of the 100-example training subset "
        f"by epoch {final_epoch}."
    )


if __name__ == "__main__":
    main()

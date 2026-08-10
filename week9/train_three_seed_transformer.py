#!/usr/bin/env python3
"""Train and diagnose the Week 9 tiny transformer with at least three seeds.

The program uses only the story text as input and predicts one of the five
carrier answers: Lammy, Anneena, Jade, Penguin, or Nobody.

Default full run, from the repository root:

    python3 train_three_seed_transformer.py

Fast pipeline check (still runs three seeds):

    python3 train_three_seed_transformer.py \
        --epochs 2 --max-train 200 --max-validation 100

Outputs are written below results/tiny_transformer_training/ and include model
settings, per-epoch losses and accuracies, final and best checkpoints, random
seeds, durations, class counts, and automatic training diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


LABELS = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
OBJECTS = ["Hairbrush", "Sneakers", "Glasses", "Key"]
SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/week6_full_dataset/train.jsonl"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("data/week6_full_dataset/validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tiny_transformer_training"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional balanced subset size; must be a multiple of 20.",
    )
    parser.add_argument(
        "--max-validation",
        type=int,
        default=None,
        help="Optional balanced subset size; must be a multiple of 20.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--overfit-gap",
        type=float,
        default=0.10,
        help="Train-minus-validation accuracy gap flagged as overfitting.",
    )
    parser.add_argument(
        "--seed-range-threshold",
        type=float,
        default=0.05,
        help="Final validation-accuracy range flagged across seeds.",
    )
    parser.add_argument(
        "--imbalance-ratio-threshold",
        type=float,
        default=1.10,
        help="Largest/smallest class count ratio considered imbalanced.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if len(args.seeds) < 3 or len(set(args.seeds)) < 3:
        raise ValueError("Provide at least three distinct random seeds")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.d_model <= 0 or args.d_model % args.heads != 0:
        raise ValueError("d-model must be positive and divisible by heads")
    if args.ff_dim <= 0:
        raise ValueError("ff-dim must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    for name in ("max_train", "max_validation"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or value % 20 != 0):
            raise ValueError(f"--{name.replace('_', '-')} must be a positive multiple of 20")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate the data with "
            "python3 scripts/generate_week6_full_dataset.py"
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("story_text"), str):
                raise ValueError(f"{path}:{line_number} has no valid story_text")
            if record.get("answer") not in LABELS:
                raise ValueError(f"{path}:{line_number} has an invalid answer")
            if record.get("question", {}).get("object") not in OBJECTS:
                raise ValueError(f"{path}:{line_number} has an invalid queried object")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def balanced_subset(
    records: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    quota = limit // (len(OBJECTS) * len(LABELS))
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {
        (object_name, label): [] for object_name in OBJECTS for label in LABELS
    }
    for record in records:
        cell = (record["question"]["object"], record["answer"])
        if len(cells[cell]) < quota:
            cells[cell].append(record)
        if all(len(examples) == quota for examples in cells.values()):
            break
    missing = {cell: quota - len(items) for cell, items in cells.items() if len(items) < quota}
    if missing:
        raise ValueError(f"Cannot create requested balanced subset: {missing}")
    return [
        cells[(object_name, label)][index]
        for index in range(quota)
        for object_name in OBJECTS
        for label in LABELS
    ]


def build_vocabulary(records: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    counts = Counter(
        token for record in records for token in tokenize(record["story_text"])
    )
    index_to_token = SPECIAL_TOKENS + sorted(counts)
    token_to_index = {token: index for index, token in enumerate(index_to_token)}
    return token_to_index, index_to_token


def encode_records(
    records: list[dict[str, Any]],
    token_to_index: dict[str, int],
    maximum_length: int,
) -> tuple[Tensor, Tensor]:
    pad_index = token_to_index["<pad>"]
    unknown_index = token_to_index["<unk>"]
    class_to_index = {label: index for index, label in enumerate(LABELS)}
    inputs = torch.full(
        (len(records), maximum_length), pad_index, dtype=torch.long
    )
    targets = torch.empty(len(records), dtype=torch.long)
    for row, record in enumerate(records):
        token_ids = [token_to_index["<cls>"]] + [
            token_to_index.get(token, unknown_index)
            for token in tokenize(record["story_text"])
        ]
        if len(token_ids) > maximum_length:
            raise ValueError("maximum_length is smaller than an encoded story")
        inputs[row, : len(token_ids)] = torch.tensor(token_ids)
        targets[row] = class_to_index[record["answer"]]
    return inputs, targets


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_projection = nn.Linear(d_model, d_model, bias=False)
        self.k_projection = nn.Linear(d_model, d_model, bias=False)
        self.v_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        batch_size, sequence_length, d_model = x.shape

        def split_heads(projected: Tensor) -> Tensor:
            return projected.view(
                batch_size, sequence_length, self.heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(self.q_projection(x))
        keys = split_heads(self.k_projection(x))
        values = split_heads(self.v_projection(x))
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = self.dropout(torch.softmax(scores, dim=-1))
        attended = attention @ values
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
        pad_index: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
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
        return self.classifier(self.final_norm(x[:, 0]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    generator: torch.Generator | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        order = torch.randperm(len(inputs), generator=generator)
    else:
        order = torch.arange(len(inputs))
    total_loss = 0.0
    total_correct = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for offset in range(0, len(inputs), batch_size):
            indices = order[offset : offset + batch_size]
            batch_inputs = inputs[indices].to(device)
            batch_targets = targets[indices].to(device)
            logits = model(batch_inputs)
            loss = loss_function(logits, batch_targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(indices)
            total_correct += logits.argmax(dim=-1).eq(batch_targets).sum().item()
    return total_loss / len(inputs), total_correct / len(inputs)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    settings: dict[str, Any],
    vocabulary: list[str],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "seed": seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "settings": settings,
        "vocabulary": vocabulary,
        "labels": LABELS,
        "history": history,
    }


def diagnose_run(
    history: list[dict[str, Any]], overfit_gap: float
) -> dict[str, Any]:
    final = history[-1]
    best = min(history, key=lambda row: row["validation_loss"])
    accuracy_gap = final["training_accuracy"] - final["validation_accuracy"]
    validation_loss_increase = (
        final["validation_loss"] / best["validation_loss"] - 1.0
        if best["validation_loss"] > 0
        else 0.0
    )
    overfitting_reasons: list[str] = []
    if accuracy_gap >= overfit_gap:
        overfitting_reasons.append(
            f"final train-validation accuracy gap is {accuracy_gap:.3f}"
        )
    if (
        best["epoch"] < final["epoch"]
        and validation_loss_increase >= 0.10
        and final["training_loss"] < best["training_loss"]
    ):
        overfitting_reasons.append(
            f"validation loss ended {validation_loss_increase:.1%} above its minimum"
        )

    instability_reasons: list[str] = []
    numeric_fields = (
        "training_loss",
        "validation_loss",
        "training_accuracy",
        "validation_accuracy",
    )
    if any(
        not math.isfinite(float(row[field]))
        for row in history
        for field in numeric_fields
    ):
        instability_reasons.append("a loss or accuracy became non-finite")
    for previous, current in zip(history, history[1:]):
        if current["training_loss"] > previous["training_loss"] * 1.50:
            instability_reasons.append(
                f"training loss jumped by more than 50% at epoch {current['epoch']}"
            )
        if current["validation_loss"] > previous["validation_loss"] * 1.50:
            instability_reasons.append(
                f"validation loss jumped by more than 50% at epoch {current['epoch']}"
            )
        if previous["validation_accuracy"] - current["validation_accuracy"] > 0.15:
            instability_reasons.append(
                f"validation accuracy dropped by more than 15 points at epoch {current['epoch']}"
            )

    return {
        "overfitting_detected": bool(overfitting_reasons),
        "overfitting_reasons": overfitting_reasons,
        "unstable_training_detected": bool(instability_reasons),
        "instability_reasons": sorted(set(instability_reasons)),
        "final_train_validation_accuracy_gap": accuracy_gap,
        "best_validation_loss": best["validation_loss"],
        "best_validation_loss_epoch": best["epoch"],
    }


def class_balance_report(
    records: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    counts = Counter(record["answer"] for record in records)
    ordered_counts = {label: counts[label] for label in LABELS}
    smallest = min(ordered_counts.values())
    largest = max(ordered_counts.values())
    ratio = largest / smallest if smallest else math.inf
    return {
        "counts": ordered_counts,
        "largest_to_smallest_ratio": ratio,
        "threshold": threshold,
        "class_imbalance_detected": ratio > threshold,
    }


def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    vocabulary: list[str],
    maximum_length: int,
    data_settings: dict[str, Any],
) -> dict[str, Any]:
    set_seed(seed)
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        **data_settings,
        "seed": seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "layers": args.layers,
        "d_model": args.d_model,
        "heads": args.heads,
        "head_dimension": args.d_model // args.heads,
        "feed_forward_dimension": args.ff_dim,
        "dropout": args.dropout,
        "maximum_sequence_length": maximum_length,
        "vocabulary_size": len(vocabulary),
        "labels": LABELS,
    }
    model = TinyTransformer(
        vocabulary_size=len(vocabulary),
        maximum_length=maximum_length,
        pad_index=0,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    settings["parameter_count"] = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.CrossEntropyLoss()
    shuffle_generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        run_epoch(
            model,
            train_inputs,
            train_targets,
            args.batch_size,
            loss_function,
            device,
            optimizer,
            shuffle_generator,
        )
        # Recompute training metrics with fixed weights and dropout disabled so
        # they are directly comparable with the validation metrics.
        training_loss, training_accuracy = run_epoch(
            model,
            train_inputs,
            train_targets,
            args.batch_size,
            loss_function,
            device,
            optimizer=None,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_inputs,
            validation_targets,
            args.batch_size,
            loss_function,
            device,
            optimizer=None,
        )
        row = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "epoch_duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"seed {seed} | epoch {epoch:02d}/{args.epochs} "
            f"| train loss {training_loss:.4f}, acc {training_accuracy:.2%} "
            f"| val loss {validation_loss:.4f}, acc {validation_accuracy:.2%}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                checkpoint_payload(
                    model, optimizer, epoch, seed, settings, vocabulary, history
                ),
                seed_dir / "checkpoint_best.pt",
            )

    duration = time.perf_counter() - started
    settings["training_duration_seconds"] = duration
    settings["final_epoch"] = args.epochs
    write_json(seed_dir / "settings.json", settings)
    write_json(seed_dir / "history.json", history)
    write_history(seed_dir / "history.csv", history)
    torch.save(
        checkpoint_payload(
            model, optimizer, args.epochs, seed, settings, vocabulary, history
        ),
        seed_dir / "checkpoint_final.pt",
    )
    diagnostics = diagnose_run(history, args.overfit_gap)
    write_json(seed_dir / "diagnostics.json", diagnostics)
    final = history[-1]
    return {
        "seed": seed,
        "training_duration_seconds": duration,
        "final_training_loss": final["training_loss"],
        "final_validation_loss": final["validation_loss"],
        "final_training_accuracy": final["training_accuracy"],
        "final_validation_accuracy": final["validation_accuracy"],
        "best_validation_accuracy": max(row["validation_accuracy"] for row in history),
        "best_validation_loss": min(row["validation_loss"] for row in history),
        "overfitting_detected": diagnostics["overfitting_detected"],
        "unstable_training_detected": diagnostics["unstable_training_detected"],
        "final_checkpoint": str((seed_dir / "checkpoint_final.pt").resolve()),
    }


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_started = time.perf_counter()

    training_records = balanced_subset(load_jsonl(args.train_data), args.max_train)
    validation_records = balanced_subset(
        load_jsonl(args.validation_data), args.max_validation
    )
    token_to_index, vocabulary = build_vocabulary(training_records)
    maximum_length = max(
        len(tokenize(record["story_text"])) + 1
        for record in training_records + validation_records
    )
    train_inputs, train_targets = encode_records(
        training_records, token_to_index, maximum_length
    )
    validation_inputs, validation_targets = encode_records(
        validation_records, token_to_index, maximum_length
    )
    training_balance = class_balance_report(
        training_records, args.imbalance_ratio_threshold
    )
    validation_balance = class_balance_report(
        validation_records, args.imbalance_ratio_threshold
    )
    balance = {"training": training_balance, "validation": validation_balance}
    write_json(args.output_dir / "class_balance.json", balance)
    write_json(args.output_dir / "vocabulary.json", vocabulary)

    data_settings = {
        "training_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "training_examples": len(training_records),
        "validation_examples": len(validation_records),
        "training_class_counts": training_balance["counts"],
        "validation_class_counts": validation_balance["counts"],
    }
    run_configuration = {
        **data_settings,
        "seeds": args.seeds,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "layers": args.layers,
        "d_model": args.d_model,
        "heads": args.heads,
        "head_dimension": args.d_model // args.heads,
        "feed_forward_dimension": args.ff_dim,
        "dropout": args.dropout,
        "vocabulary_size": len(vocabulary),
        "maximum_sequence_length": maximum_length,
        "overfit_accuracy_gap_threshold": args.overfit_gap,
        "class_imbalance_ratio_threshold": args.imbalance_ratio_threshold,
        "seed_accuracy_range_threshold": args.seed_range_threshold,
    }
    write_json(args.output_dir / "run_configuration.json", run_configuration)
    print("Three-seed tiny-transformer experiment")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(
        f"Examples: {len(training_records):,} train, "
        f"{len(validation_records):,} validation"
    )
    print(f"Vocabulary: {len(vocabulary)} | maximum length: {maximum_length}")

    seed_results = [
        train_one_seed(
            seed,
            args,
            device,
            train_inputs,
            train_targets,
            validation_inputs,
            validation_targets,
            vocabulary,
            maximum_length,
            data_settings,
        )
        for seed in args.seeds
    ]
    final_accuracies = [row["final_validation_accuracy"] for row in seed_results]
    seed_range = max(final_accuracies) - min(final_accuracies)
    aggregate_diagnostics = {
        "overfitting_detected_in_any_seed": any(
            row["overfitting_detected"] for row in seed_results
        ),
        "unstable_training_detected_in_any_seed": any(
            row["unstable_training_detected"] for row in seed_results
        ),
        "class_imbalance_detected": (
            training_balance["class_imbalance_detected"]
            or validation_balance["class_imbalance_detected"]
        ),
        "final_validation_accuracy_range": seed_range,
        "final_validation_accuracy_mean": statistics.mean(final_accuracies),
        "final_validation_accuracy_population_std": statistics.pstdev(
            final_accuracies
        ),
        "final_validation_accuracy_minimum": min(final_accuracies),
        "final_validation_accuracy_maximum": max(final_accuracies),
        "seed_difference_threshold": args.seed_range_threshold,
        "large_random_seed_difference_detected": (
            seed_range >= args.seed_range_threshold
        ),
        "total_experiment_duration_seconds": time.perf_counter() - all_started,
    }
    summary = {
        "model": "tiny_word_level_transformer",
        "seeds": args.seeds,
        "seed_results": seed_results,
        "class_balance": balance,
        "diagnostics": aggregate_diagnostics,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "diagnostics.json", aggregate_diagnostics)
    write_summary_csv(args.output_dir / "seed_summary.csv", seed_results)

    print("\nExperiment diagnostics")
    print(json.dumps(aggregate_diagnostics, indent=2))
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

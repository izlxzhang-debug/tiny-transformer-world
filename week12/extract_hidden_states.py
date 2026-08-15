#!/usr/bin/env python3
"""Save hidden states from every tiny-transformer layer for future probes.

This program deliberately does not run PCA, train probes, or interpret any
representation. It loads one trained Week 9 checkpoint, runs a chosen dataset
split through the frozen transformer, and saves:

* the embedding-stage representation;
* the output of every transformer layer;
* the final layer-normalized representation;
* token IDs and padding masks; and
* raw targets for five planned probe tasks.

Default run from the repository root:

    python3 week12/extract_hidden_states.py

Small verification run:

    python3 week12/extract_hidden_states.py --device cpu --max-examples 40

By default only the <cls> vector is saved at each layer. Use
--representation full to save every token representation; this requires much
more disk space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


LABELS = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
AGENTS = ["Lammy", "Anneena", "Jade", "Penguin"]
OBJECTS = ["Hairbrush", "Sneakers", "Glasses", "Key"]
LOCATIONS = ["Mars", "Mercury", "Venus", "Moon"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/tiny_transformer_training/seed_11/checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/week6_full_dataset/validation.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hidden_states/seed_11_validation.pt"),
    )
    parser.add_argument(
        "--probe-plan",
        type=Path,
        default=Path("results/hidden_states/probe_plan.json"),
    )
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=Path("results/hidden_states/seed_11_validation_targets.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--representation",
        choices=("cls", "full"),
        default="cls",
        help="Save one <cls> vector per story or every token vector.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float16"),
        default="float32",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")


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


def load_jsonl(path: Path, maximum: int | None) -> list[dict[str, Any]]:
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
            required = {
                "story_id",
                "story_text",
                "initial_state",
                "events",
                "question",
                "answer",
                "metrics",
                "evidence_event_ids",
                "final_state",
            }
            if not required.issubset(record):
                raise ValueError(f"{path}:{line_number} has an invalid record")
            if record["answer"] not in LABELS:
                raise ValueError(f"{path}:{line_number} has an invalid answer")
            records.append(record)
            if maximum is not None and len(records) == maximum:
                break
    if not records:
        raise ValueError(f"{path} contains no examples")
    return records


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
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


class TinyTransformerWithHiddenStates(nn.Module):
    """Checkpoint-compatible transformer that can return every hidden state."""

    def __init__(
        self,
        vocabulary_size: int,
        position_capacity: int,
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
        self.position_embedding = nn.Embedding(position_capacity, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, ff_dim, dropout)
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, len(LABELS))

    def forward(
        self, token_ids: Tensor, return_hidden_states: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the checkpoint's "
                f"learned position capacity "
                f"{self.position_embedding.num_embeddings}. Do not truncate "
                "interpretability examples silently."
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        hidden_states: dict[str, Tensor] = {"embedding": x}
        padding_mask = token_ids.eq(self.pad_index)
        for layer_number, block in enumerate(self.blocks, start=1):
            x = block(x, padding_mask)
            hidden_states[f"layer_{layer_number}"] = x
        normalized = self.final_norm(x)
        hidden_states["final_norm"] = normalized
        logits = self.classifier(normalized[:, 0])
        if return_hidden_states:
            return logits, hidden_states
        return logits


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TinyTransformerWithHiddenStates, dict[str, int], dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required = {"model_state", "settings", "vocabulary", "labels", "seed", "epoch"}
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint has an incompatible format")
    if checkpoint["labels"] != LABELS:
        raise ValueError("Checkpoint label order does not match this program")
    settings = checkpoint["settings"]
    vocabulary = checkpoint["vocabulary"]
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    position_capacity = checkpoint["model_state"][
        "position_embedding.weight"
    ].shape[0]
    model = TinyTransformerWithHiddenStates(
        vocabulary_size=len(vocabulary),
        position_capacity=position_capacity,
        pad_index=token_to_index["<pad>"],
        d_model=int(settings["d_model"]),
        heads=int(settings["heads"]),
        layers=int(settings["layers"]),
        ff_dim=int(settings["feed_forward_dimension"]),
        dropout=float(settings["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "seed": int(checkpoint["seed"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "d_model": int(settings["d_model"]),
        "layers": int(settings["layers"]),
        "heads": int(settings["heads"]),
        "position_capacity": int(position_capacity),
        "vocabulary_size": len(vocabulary),
    }
    return model, token_to_index, metadata


def encode_records(
    records: list[dict[str, Any]], token_to_index: dict[str, int]
) -> tuple[Tensor, Tensor, Tensor]:
    pad_index = token_to_index["<pad>"]
    unknown_index = token_to_index["<unk>"]
    encoded = [
        [token_to_index["<cls>"]]
        + [
            token_to_index.get(token, unknown_index)
            for token in tokenize(record["story_text"])
        ]
        for record in records
    ]
    maximum_length = max(map(len, encoded))
    inputs = torch.full(
        (len(records), maximum_length), pad_index, dtype=torch.long
    )
    lengths = torch.tensor([len(token_ids) for token_ids in encoded], dtype=torch.long)
    for row, token_ids in enumerate(encoded):
        inputs[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    attention_mask = inputs.ne(pad_index)
    return inputs, attention_mask, lengths


def effective_object_location(record: dict[str, Any]) -> str:
    queried_object = record["question"]["object"]
    value = record["final_state"]["object_states"][queried_object]
    if value in AGENTS:
        return record["final_state"]["agent_locations"][value]
    if value not in LOCATIONS:
        raise ValueError(f"Invalid final object state: {value}")
    return value


def contains_distractor(record: dict[str, Any]) -> bool:
    evidence_ids = set(record["evidence_event_ids"])
    return any(event["event_id"] not in evidence_ids for event in record["events"])


def build_probe_targets(records: list[dict[str, Any]]) -> dict[str, Tensor]:
    location_to_index = {label: index for index, label in enumerate(LOCATIONS)}
    carrier_to_index = {label: index for index, label in enumerate(LABELS)}
    return {
        "object_location": torch.tensor(
            [location_to_index[effective_object_location(record)] for record in records],
            dtype=torch.long,
        ),
        "is_carried": torch.tensor(
            [int(record["answer"] != "Nobody") for record in records],
            dtype=torch.long,
        ),
        "current_carrier": torch.tensor(
            [carrier_to_index[record["answer"]] for record in records],
            dtype=torch.long,
        ),
        "story_length": torch.tensor(
            [int(record["metrics"]["story_length"]) for record in records],
            dtype=torch.long,
        ),
        "has_distractor": torch.tensor(
            [int(contains_distractor(record)) for record in records],
            dtype=torch.long,
        ),
    }


def probe_plan() -> dict[str, Any]:
    return {
        "status": "planned_not_run",
        "pca_status": "not_run_and_not_interpreted",
        "representation": (
            "Train a separate linear probe on the frozen <cls> vector from each "
            "saved layer. Do not update transformer weights."
        ),
        "tasks": {
            "object_location": {
                "type": "four_class_classification",
                "labels": LOCATIONS,
                "target": "Final effective location of the queried object.",
                "metrics": ["accuracy", "macro_f1"],
            },
            "is_carried": {
                "type": "binary_classification",
                "labels": ["not_carried", "carried"],
                "target": "Whether the queried object has a current carrier.",
                "metrics": ["balanced_accuracy", "roc_auc"],
            },
            "current_carrier": {
                "type": "five_class_classification",
                "labels": LABELS,
                "target": "Current carrier of the queried object, including Nobody.",
                "metrics": ["accuracy", "macro_f1"],
            },
            "story_length": {
                "type": "regression",
                "target": "Number of ordered events in the story.",
                "metrics": ["mean_absolute_error", "r_squared"],
            },
            "has_distractor": {
                "type": "binary_classification",
                "labels": ["no_distractor", "has_distractor"],
                "target": (
                    "Whether any event is outside the queried object's "
                    "evidence_event_ids."
                ),
                "metrics": ["balanced_accuracy", "roc_auc"],
            },
        },
        "protocol": {
            "fit_split": "train hidden states",
            "model_selection_split": "validation hidden states",
            "final_report_split": "test_standard hidden states",
            "standardization": "Fit feature scaling on probe-training data only.",
            "regularization": (
                "Choose linear-probe regularization on validation only and use "
                "the same candidate values for every layer."
            ),
            "controls": [
                "majority or mean-target baseline",
                "random-label control",
                "report target class counts",
                "repeat probe fitting with at least three probe seeds",
                "compare identical train/validation/test examples across layers",
            ],
            "interpretation_limit": (
                "Probe decodability does not by itself show that the transformer "
                "uses the decoded feature causally."
            ),
        },
    }


def write_probe_targets_csv(
    path: Path, records: list[dict[str, Any]], targets: dict[str, Tensor]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "story_id",
            "queried_object",
            "object_location",
            "object_location_index",
            "is_carried",
            "current_carrier",
            "current_carrier_index",
            "story_length",
            "has_distractor",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "story_id": record["story_id"],
                    "queried_object": record["question"]["object"],
                    "object_location": effective_object_location(record),
                    "object_location_index": targets["object_location"][index].item(),
                    "is_carried": targets["is_carried"][index].item(),
                    "current_carrier": record["answer"],
                    "current_carrier_index": targets["current_carrier"][index].item(),
                    "story_length": targets["story_length"][index].item(),
                    "has_distractor": targets["has_distractor"][index].item(),
                }
            )


@torch.no_grad()
def extract_hidden_states(
    model: TinyTransformerWithHiddenStates,
    inputs: Tensor,
    batch_size: int,
    device: torch.device,
    representation: str,
    storage_dtype: torch.dtype,
) -> tuple[dict[str, Tensor], Tensor]:
    collected: dict[str, list[Tensor]] = {}
    logits_parts: list[Tensor] = []
    for offset in range(0, len(inputs), batch_size):
        batch = inputs[offset : offset + batch_size].to(device)
        logits, states = model(batch, return_hidden_states=True)
        logits_parts.append(logits.cpu())
        for layer_name, state in states.items():
            selected = state[:, 0] if representation == "cls" else state
            collected.setdefault(layer_name, []).append(
                selected.detach().to(device="cpu", dtype=storage_dtype)
            )
    return (
        {name: torch.cat(parts, dim=0) for name, parts in collected.items()},
        torch.cat(logits_parts, dim=0),
    )


def class_counts(values: Tensor, labels: list[str] | None = None) -> dict[str, int]:
    counts = Counter(values.tolist())
    if labels is None:
        return {str(key): counts[key] for key in sorted(counts)}
    return {label: counts[index] for index, label in enumerate(labels)}


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    records = load_jsonl(args.data, args.max_examples)
    model, token_to_index, checkpoint_metadata = load_model(args.checkpoint, device)
    inputs, attention_mask, sequence_lengths = encode_records(records, token_to_index)
    if inputs.shape[1] > checkpoint_metadata["position_capacity"]:
        raise ValueError(
            f"Dataset needs {inputs.shape[1]} positions, but the checkpoint learned "
            f"only {checkpoint_metadata['position_capacity']}. Use a compatible "
            "split rather than silently truncating interpretability data."
        )
    storage_dtype = torch.float32 if args.storage_dtype == "float32" else torch.float16
    hidden_states, logits = extract_hidden_states(
        model,
        inputs,
        args.batch_size,
        device,
        args.representation,
        storage_dtype,
    )
    targets = build_probe_targets(records)
    predictions = logits.argmax(dim=-1)
    carrier_targets = targets["current_carrier"]
    classifier_accuracy = predictions.eq(carrier_targets).float().mean().item()
    plan = probe_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            **checkpoint_metadata,
            "data": str(args.data.resolve()),
            "examples": len(records),
            "representation": args.representation,
            "storage_dtype": args.storage_dtype,
            "layer_names": list(hidden_states),
            "hidden_shapes": {
                name: list(value.shape) for name, value in hidden_states.items()
            },
            "maximum_sequence_length": int(inputs.shape[1]),
            "classifier_accuracy_on_extracted_examples": classifier_accuracy,
            "pca_performed": False,
            "probe_training_performed": False,
            "target_counts": {
                "object_location": class_counts(targets["object_location"], LOCATIONS),
                "is_carried": class_counts(
                    targets["is_carried"], ["not_carried", "carried"]
                ),
                "current_carrier": class_counts(
                    targets["current_carrier"], LABELS
                ),
                "story_length": class_counts(targets["story_length"]),
                "has_distractor": class_counts(
                    targets["has_distractor"], ["no_distractor", "has_distractor"]
                ),
            },
        },
        "hidden_states": hidden_states,
        "token_ids": inputs,
        "attention_mask": attention_mask,
        "sequence_lengths": sequence_lengths,
        "story_ids": [record["story_id"] for record in records],
        "queried_objects": [record["question"]["object"] for record in records],
        "probe_targets": targets,
        "probe_label_maps": {
            "object_location": LOCATIONS,
            "is_carried": ["not_carried", "carried"],
            "current_carrier": LABELS,
            "has_distractor": ["no_distractor", "has_distractor"],
        },
        "classifier_logits": logits,
    }
    torch.save(payload, args.output)
    args.probe_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.probe_plan.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
    write_probe_targets_csv(args.targets_csv, records, targets)
    print("Hidden-state extraction complete")
    print(f"Examples: {len(records):,}")
    print(f"Representation: {args.representation}")
    for layer_name, values in hidden_states.items():
        print(f"{layer_name:12s}: {list(values.shape)}")
    print(f"Classifier accuracy: {classifier_accuracy:.2%}")
    print(f"Hidden states: {args.output.resolve()}")
    print(f"Probe plan: {args.probe_plan.resolve()}")
    print(f"Targets CSV: {args.targets_csv.resolve()}")
    print("PCA performed: no")
    print("Probe training performed: no")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract hidden vectors and analyze them with PCA and controlled probes.

This one file contains the model/checkpoint reader, extraction, PCA, linear
and optional nonlinear probes, and the original comparison controls. A trained
model checkpoint and the story dataset are required inputs.

Default run from the repository root:

    python3 'Training and Testing Transformer/Looking Inside the Model/look_inside.py'

Small verification run:

    python3 'Training and Testing Transformer/Looking Inside the Model/look_inside.py' --device cpu --max-examples 200

The default run extracts train, validation, and standard-test vectors and then
runs the probes. Use 'extract --help' or 'probe --help' to run either stage
separately. Only the extraction stage supports --representation full; probes
expect one <cls> vector per example. Raw artifacts stay in results/, while
combined measurements are written to the adjacent results.json.
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


def parse_extract_args() -> argparse.Namespace:
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
        default=Path("Dataset/Training and Harder Tests/dataset.json"),
    )
    parser.add_argument(
        "--split", default="validation",
        choices=("train", "validation", "test_standard", "test_long",
                 "test_paraphrase", "test_withheld"),
        help="Named split to read when --data is a packaged dataset.json.",
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
    parser.add_argument("--results-file", type=Path,
        default=Path(__file__).resolve().with_name("results.json"),
        help="Combined JSON results written beside this program by default.")
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


def iter_dataset_records(path: Path, split: str | None = None):
    """Read an old split file or a named split from the Week 6 JSON bundle."""
    selected_split = split or path.stem
    if path.is_file():
        source = path
    else:
        source = path.parent / "dataset.json"
        if not source.is_file():
            root = next(
                (parent for parent in Path(__file__).resolve().parents
                 if (parent / "Dataset").is_dir()),
                Path.cwd(),
            )
            allowed = {root / "data/week6_full_dataset", root / "Dataset/Training and Harder Tests"}
            source = root / "Dataset/Training and Harder Tests/dataset.json"
            if path.parent.resolve() not in allowed or not source.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Run python3 'Dataset/Training and Harder Tests/generate_full_dataset.py'."
                )
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix == ".json":
            bundle = json.load(handle)
            if selected_split not in bundle["splits"]:
                raise ValueError(f"Unknown dataset split: {selected_split}")
            yield from bundle["splits"][selected_split]
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_jsonl(
    path: Path, maximum: int | None, split: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, record in enumerate(iter_dataset_records(path, split), start=1):
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


def extract_main() -> None:
    args = parse_extract_args()
    validate_args(args)
    device = choose_device(args.device)
    selected_split = args.split if args.data.suffix == ".json" else args.data.stem
    records = load_jsonl(args.data, args.max_examples, split=selected_split)
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
            "data_split": selected_split,
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
    write_results_bundle(
        [args.output.parent], args.results_file, "Looking Inside the Model",
        notes=["Hidden vectors were extracted; this stage does not train probes.",
               "Use the probe stage or the default combined run to analyze them."],
    )


# PCA, probes, and control experiments

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import sparse
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing experiment packages. Install them with:\n"
        "  python3 -m pip install numpy scipy scikit-learn matplotlib\n"
        f"Original error: {exc}"
    )


TASKS = {
    "object_location": "classification",
    "is_carried": "classification",
    "current_carrier": "classification",
    "story_length": "regression",
    "has_distractor": "classification",
}
PROBE_SEEDS = (11, 22, 33)
LINEAR_GRID = (0.01, 0.1, 1.0, 10.0)
MLP_GRID = (0.0001, 0.001, 0.01)


def parse_probe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("results/hidden_states/train.pt"))
    parser.add_argument(
        "--validation", type=Path, default=Path("results/hidden_states/validation.pt")
    )
    parser.add_argument(
        "--test", type=Path, default=Path("results/hidden_states/test_standard.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/probe_experiments")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        help="Also run one-hidden-layer MLP probes (slower).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional training subset for a quick smoke test.",
    )
    parser.add_argument("--results-file", type=Path,
        default=Path(__file__).resolve().with_name("results.json"),
        help="Combined JSON results written beside this program by default.")
    return parser.parse_args()


def torch_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def stable_offset(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def validate_splits(splits: dict[str, dict[str, Any]]) -> list[str]:
    layer_names = list(splits["train"]["hidden_states"])
    target_names = set(TASKS)
    for split_name, payload in splits.items():
        if payload["metadata"].get("representation") != "cls":
            raise ValueError(f"{split_name} must contain CLS representations")
        if list(payload["hidden_states"]) != layer_names:
            raise ValueError("All splits must contain the same layers")
        if not target_names.issubset(payload["probe_targets"]):
            raise ValueError(f"{split_name} is missing probe targets")
        n = len(payload["story_ids"])
        if any(len(x) != n for x in payload["hidden_states"].values()):
            raise ValueError(f"Row-count mismatch in {split_name}")
    id_sets = {name: set(value["story_ids"]) for name, value in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            raise ValueError(f"Data leakage: {len(overlap)} story IDs overlap in {left}/{right}")
    return layer_names


def subset_train(payload: dict[str, Any], maximum: int | None) -> dict[str, Any]:
    if maximum is None or maximum >= len(payload["story_ids"]):
        return payload
    if maximum <= 0:
        raise ValueError("--max-train must be positive")
    rng = np.random.default_rng(2026)
    keep = np.sort(rng.choice(len(payload["story_ids"]), size=maximum, replace=False))
    copy = dict(payload)
    copy["story_ids"] = [payload["story_ids"][i] for i in keep]
    copy["hidden_states"] = {k: v[keep] for k, v in payload["hidden_states"].items()}
    copy["probe_targets"] = {k: v[keep] for k, v in payload["probe_targets"].items()}
    copy["token_ids"] = payload["token_ids"][keep]
    if "attention_mask" in payload:
        copy["attention_mask"] = payload["attention_mask"][keep]
    return copy


def numpy_targets(payload: dict[str, Any], task: str) -> np.ndarray:
    values = payload["probe_targets"][task].detach().cpu().numpy()
    return values.astype(np.float64 if TASKS[task] == "regression" else np.int64)


def randomize_targets(y: np.ndarray, seed: int, task: str, split: str) -> np.ndarray:
    rng = np.random.default_rng(seed + stable_offset(task + ":" + split))
    return y[rng.permutation(len(y))]


def classification_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    classes = np.arange(prob.shape[1])
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(y, prob, labels=classes)),
    }
    if prob.shape[1] == 2 and len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, prob[:, 1]))
    return result


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    correlation = spearmanr(y, pred).statistic
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(correlation) if np.isfinite(correlation) else float("nan"),
    }


def fit_classification(
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    y_val: np.ndarray,
    seed: int,
    probe_type: str,
    sparse_input: bool,
) -> tuple[Any, float]:
    best_model, best_value, best_score = None, None, -np.inf
    grid = LINEAR_GRID if probe_type == "linear" else MLP_GRID
    for value in grid:
        if probe_type == "linear":
            estimator = LogisticRegression(
                C=value, max_iter=2000, solver="lbfgs", random_state=seed
            )
        else:
            estimator = MLPClassifier(
                hidden_layer_sizes=(64,), activation="relu", alpha=value,
                early_stopping=False, max_iter=300, random_state=seed,
            )
        model = make_pipeline(StandardScaler(with_mean=not sparse_input), estimator)
        model.fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
        if score > best_score:
            best_model, best_value, best_score = model, value, score
    return best_model, float(best_value)


def fit_regression(
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    y_val: np.ndarray,
    seed: int,
    probe_type: str,
    sparse_input: bool,
) -> tuple[Any, float, float, float]:
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    train_z = (y_train - y_mean) / y_scale
    grid = LINEAR_GRID if probe_type == "linear" else MLP_GRID
    best_model, best_value, best_score = None, None, np.inf
    for value in grid:
        if probe_type == "linear":
            estimator = Ridge(alpha=value)
        else:
            estimator = MLPRegressor(
                hidden_layer_sizes=(64,), activation="relu", alpha=value,
                early_stopping=False, max_iter=300, random_state=seed,
            )
        model = make_pipeline(StandardScaler(with_mean=not sparse_input), estimator)
        model.fit(x_train, train_z)
        prediction = model.predict(x_val) * y_scale + y_mean
        score = np.sqrt(mean_squared_error(y_val, prediction))
        if score < best_score:
            best_model, best_value, best_score = model, value, score
    return best_model, float(best_value), y_mean, y_scale


def run_probe(
    features: dict[str, Any],
    targets: dict[str, np.ndarray],
    task: str,
    seed: int,
    probe_type: str,
    sparse_input: bool = False,
) -> dict[str, Any]:
    if TASKS[task] == "classification":
        model, selected = fit_classification(
            features["train"], targets["train"], features["validation"],
            targets["validation"], seed, probe_type, sparse_input,
        )
        pred = model.predict(features["test"])
        prob = model.predict_proba(features["test"])
        metrics = classification_metrics(targets["test"], pred, prob)
    else:
        model, selected, y_mean, y_scale = fit_regression(
            features["train"], targets["train"], features["validation"],
            targets["validation"], seed, probe_type, sparse_input,
        )
        pred = model.predict(features["test"]) * y_scale + y_mean
        metrics = regression_metrics(targets["test"], pred)
    return {"selected_regularization": selected, **metrics}


def word_count_features(
    payload: dict[str, Any], vocabulary_size: int, excluded: set[int]
) -> sparse.csr_matrix:
    rows, columns, values = [], [], []
    for row, token_row in enumerate(payload["token_ids"].tolist()):
        counts = Counter(token for token in token_row if token not in excluded)
        for token, count in counts.items():
            rows.append(row)
            columns.append(token)
            values.append(count)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(payload["story_ids"]), vocabulary_size),
        dtype=np.float32,
    )


@torch.no_grad()
def untrained_hidden_states(
    splits: dict[str, dict[str, Any]], seed: int, batch_size: int
) -> dict[str, dict[str, np.ndarray]]:
    checkpoint_path = Path(splits["train"]["metadata"]["checkpoint"])
    checkpoint = torch_load(checkpoint_path)
    settings = checkpoint["settings"]
    vocabulary = checkpoint["vocabulary"]
    token_to_index = {token: i for i, token in enumerate(vocabulary)}
    position_capacity = checkpoint["model_state"]["position_embedding.weight"].shape[0]
    torch.manual_seed(seed)
    model = TinyTransformerWithHiddenStates(
        vocabulary_size=len(vocabulary),
        position_capacity=position_capacity,
        pad_index=token_to_index["<pad>"],
        d_model=int(settings["d_model"]),
        heads=int(settings["heads"]),
        layers=int(settings["layers"]),
        ff_dim=int(settings["feed_forward_dimension"]),
        dropout=float(settings["dropout"]),
    ).eval()
    result: dict[str, dict[str, np.ndarray]] = {}
    for split_name, payload in splits.items():
        inputs = payload["token_ids"]
        if inputs.shape[1] > position_capacity:
            raise ValueError("An input exceeds the untrained model's position capacity")
        parts: dict[str, list[np.ndarray]] = defaultdict(list)
        for start in range(0, len(inputs), batch_size):
            _, states = model(inputs[start : start + batch_size], return_hidden_states=True)
            for layer, state in states.items():
                parts[layer].append(state[:, 0].cpu().numpy().astype(np.float32))
        result[split_name] = {layer: np.concatenate(x) for layer, x in parts.items()}
    return result


def run_pca(
    splits: dict[str, dict[str, Any]], layers: list[str], output_dir: Path
) -> list[dict[str, Any]]:
    plot_dir = output_dir / "pca"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = numpy_targets(splits["test"], "current_carrier").astype(int)
    rows = []
    for layer in layers:
        train_x = splits["train"]["hidden_states"][layer].float().numpy()
        test_x = splits["test"]["hidden_states"][layer].float().numpy()
        scaler = StandardScaler().fit(train_x)
        pca = PCA(n_components=2, random_state=0).fit(scaler.transform(train_x))
        points = pca.transform(scaler.transform(test_x))
        rows.append({
            "layer": layer,
            "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
            "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
        })
        plt.figure(figsize=(7, 5))
        for class_index, label in enumerate(LABELS):
            mask = labels == class_index
            plt.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.45, label=label)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA of test CLS states: {layer}")
        plt.legend(markerscale=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{layer}.png", dpi=160)
        plt.close()
    return rows


def add_result(
    rows: list[dict[str, Any]], condition: str, representation: str, task: str,
    probe_type: str, seed: int, elapsed: float, result: dict[str, Any]
) -> None:
    rows.append({
        "condition": condition,
        "representation": representation,
        "task": task,
        "probe_type": probe_type,
        "seed": seed,
        "seconds": elapsed,
        **result,
    })


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = ("condition", "representation", "task", "probe_type")
    metric_names = (
        "accuracy", "balanced_accuracy", "macro_f1", "cross_entropy", "auroc",
        "mae", "rmse", "r2", "spearman",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in identity)].append(row)
    output = []
    for key, group in groups.items():
        item = dict(zip(identity, key))
        item["runs"] = len(group)
        for metric in metric_names:
            values = [float(row[metric]) for row in group if metric in row]
            if values:
                item[f"{metric}_mean"] = float(np.nanmean(values))
                item[f"{metric}_std"] = float(np.nanstd(values))
        output.append(item)
    return output


def probe_main() -> None:
    args = parse_probe_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": torch_load(args.train),
        "validation": torch_load(args.validation),
        "test": torch_load(args.test),
    }
    splits["train"] = subset_train(splits["train"], args.max_train)
    layers = validate_splits(splits)
    targets = {
        task: {name: numpy_targets(payload, task) for name, payload in splits.items()}
        for task in TASKS
    }

    pca_rows = run_pca(splits, layers, args.output_dir)
    write_csv(args.output_dir / "pca_explained_variance.csv", pca_rows)

    counts = {
        split: {
            task: {str(k): int(v) for k, v in Counter(values[split].tolist()).items()}
            for task, values in targets.items()
        }
        for split in splits
    }
    (args.output_dir / "target_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )

    checkpoint = torch_load(Path(splits["train"]["metadata"]["checkpoint"]))
    vocabulary = checkpoint["vocabulary"]
    excluded = {vocabulary.index("<pad>"), vocabulary.index("<cls>")}
    bow = {
        name: word_count_features(payload, len(vocabulary), excluded)
        for name, payload in splits.items()
    }

    rows: list[dict[str, Any]] = []
    probe_types = ["linear"] + (["nonlinear"] if args.nonlinear else [])
    for seed in PROBE_SEEDS:
        for layer in layers:
            features = {
                name: payload["hidden_states"][layer].float().numpy()
                for name, payload in splits.items()
            }
            for task in TASKS:
                for probe_type in probe_types:
                    start = time.perf_counter()
                    result = run_probe(features, targets[task], task, seed, probe_type)
                    add_result(
                        rows, "trained_transformer", layer, task, probe_type,
                        seed, time.perf_counter() - start, result,
                    )

                randomized = {
                    split: randomize_targets(values, seed, task, split)
                    for split, values in targets[task].items()
                }
                start = time.perf_counter()
                result = run_probe(features, randomized, task, seed, "linear")
                add_result(
                    rows, "random_labels", layer, task, "linear", seed,
                    time.perf_counter() - start, result,
                )

        for task in TASKS:
            start = time.perf_counter()
            result = run_probe(bow, targets[task], task, seed, "linear", sparse_input=True)
            add_result(
                rows, "input_word_counts", "bag_of_words", task, "linear", seed,
                time.perf_counter() - start, result,
            )

        length_only = {
            name: targets["story_length"][name].reshape(-1, 1)
            for name in splits
        }
        start = time.perf_counter()
        result = run_probe(
            length_only, targets["has_distractor"], "has_distractor", seed, "linear"
        )
        add_result(
            rows, "length_only_baseline", "event_count", "has_distractor",
            "linear", seed, time.perf_counter() - start, result,
        )

        untrained = untrained_hidden_states(splits, seed, args.batch_size)
        for layer in layers:
            features = {name: untrained[name][layer] for name in splits}
            for task in TASKS:
                start = time.perf_counter()
                result = run_probe(features, targets[task], task, seed, "linear")
                add_result(
                    rows, "untrained_transformer", layer, task, "linear", seed,
                    time.perf_counter() - start, result,
                )

    write_csv(args.output_dir / "probe_runs.csv", rows)
    summary = summarize(rows)
    write_csv(args.output_dir / "probe_summary.csv", summary)
    metadata = {
        "train": str(args.train.resolve()),
        "validation": str(args.validation.resolve()),
        "test": str(args.test.resolve()),
        "layers": layers,
        "probe_seeds": list(PROBE_SEEDS),
        "nonlinear_enabled": args.nonlinear,
        "max_train": args.max_train,
        "notes": {
            "current_carrier": "Sanity/readout probe: this is the original training target.",
            "has_distractor": (
                "Exploratory: extractor labels use evidence_event_ids and may be "
                "confounded with story length. Compare against story_length results."
            ),
            "pca": "PCA was fit on standardized training states and applied to test states.",
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Finished {len(rows)} probe runs")
    print(f"Run-level results: {(args.output_dir / 'probe_runs.csv').resolve()}")
    print(f"Mean/std summary: {(args.output_dir / 'probe_summary.csv').resolve()}")
    print(f"PCA plots: {(args.output_dir / 'pca').resolve()}")
    write_results_bundle([args.output_dir], args.results_file, "Looking Inside the Model")


def write_results_bundle(directories, destination, experiment, *, notes=None):
    """Save measured JSON/CSV results together; reference large binary files."""
    import csv
    files = {}
    artifacts = []
    hidden_state_metadata = {}
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for source in sorted(directory.rglob("*")):
            if not source.is_file() or source.resolve() == Path(destination).resolve():
                continue
            if any("smoke" in part for part in source.relative_to(directory).parts):
                continue
            key = str(source)
            if source.suffix == ".json":
                files[key] = json.loads(source.read_text(encoding="utf-8"))
            elif source.suffix == ".csv":
                with source.open(encoding="utf-8", newline="") as handle:
                    files[key] = list(csv.DictReader(handle))
            elif source.suffix in {".pt", ".png", ".npy"}:
                artifacts.append({"path": key, "bytes": source.stat().st_size})
                if source.name in {"train.pt", "validation.pt", "test_standard.pt"}:
                    import torch
                    saved = torch.load(source, map_location="cpu", weights_only=False)
                    if "metadata" in saved:
                        hidden_state_metadata[source.stem] = saved["metadata"]
    payload = {
        "experiment": experiment,
        "source_directories": [str(Path(directory)) for directory in directories],
        "files": files,
        "artifacts": artifacts,
        "hidden_state_metadata": hidden_state_metadata,
        "notes": notes or [
            "Results were collected from the source files named below; no scores are inferred.",
            "CSV values are preserved as text; JSON reports retain their original types.",
            "Checkpoints, hidden vectors, and images are referenced, not embedded.",
            "Use the recorded configuration to distinguish full runs from quick checks.",
        ],
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved combined results to {destination}")
    return payload


def run_with_arguments(function, arguments):
    """Call a stage with its own CLI arguments and then restore the caller."""
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        return function()
    finally:
        sys.argv = previous


def main():
    """Extract three distinct datasets, then analyze their frozen vectors."""
    arguments = sys.argv[1:]
    if arguments[:1] == ["extract"]:
        run_with_arguments(extract_main, arguments[1:])
        return
    if arguments[:1] == ["probe"]:
        run_with_arguments(probe_main, arguments[1:])
        return
    parser = argparse.ArgumentParser(
        description="Extract hidden vectors and run PCA and controlled probes. "
                    "Use 'extract --help' or 'probe --help' for individual stages.",
    )
    parser.add_argument("--checkpoint", type=Path,
        default=Path("results/tiny_transformer_training/seed_11/checkpoint_best.pt"))
    parser.add_argument("--data", type=Path,
        default=Path("Dataset/Training and Harder Tests/dataset.json"))
    parser.add_argument("--hidden-dir", type=Path, default=Path("results/hidden_states"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/probe_experiments"))
    parser.add_argument("--results-file", type=Path,
        default=Path(__file__).resolve().with_name("results.json"))
    parser.add_argument("--max-examples", type=int, default=None,
        help="Optional limit for each extracted split; used only for quick checks.")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--nonlinear", action="store_true")
    args = parser.parse_args()
    args.hidden_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test_standard"):
        command = [
            "--checkpoint", str(args.checkpoint), "--data", str(args.data),
            "--split", split, "--output", str(args.hidden_dir / f"{split}.pt"),
            "--probe-plan", str(args.hidden_dir / f"{split}_plan.json"),
            "--targets-csv", str(args.hidden_dir / f"{split}_targets.csv"),
            "--batch-size", str(args.batch_size), "--device", args.device,
            "--results-file", str(args.results_file),
        ]
        if args.max_examples is not None:
            command += ["--max-examples", str(args.max_examples)]
        run_with_arguments(extract_main, command)
    command = [
        "--train", str(args.hidden_dir / "train.pt"),
        "--validation", str(args.hidden_dir / "validation.pt"),
        "--test", str(args.hidden_dir / "test_standard.pt"),
        "--output-dir", str(args.output_dir),
        "--batch-size", str(args.batch_size),
        "--results-file", str(args.results_file),
    ]
    if args.max_train is not None:
        command += ["--max-train", str(args.max_train)]
    if args.nonlinear:
        command.append("--nonlinear")
    run_with_arguments(probe_main, command)
    write_results_bundle(
        [args.hidden_dir, args.output_dir], args.results_file,
        "Looking Inside the Model",
        notes=[
            "Extraction and probes ran together using the three named data splits.",
            "Probe performance measures recoverability, not causal use or consciousness.",
            "Binary hidden vectors and plots remain at their recorded paths.",
        ],
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate all trained transformer seeds and baselines on one shared table.

Run from the repository root:

    python3 week10/evaluate_generalization.py

The evaluator covers standard stories, longer stories, withheld pickup
combinations, unseen character names, stories with distractor events, and
paraphrases. It compares every transformer seed with a majority baseline and
a bag-of-words logistic-regression baseline, then writes CSV, JSON, and
Markdown tables under results/generalization_evaluation/.

The trained checkpoints contain 142 learned positional embeddings, while the
long split can reach 179 tokens. By default, extra positions repeat the final
learned position vector. This preserves every story token without pretending
that the model learned positions it never saw. The output metadata records the
extension. Use --position-extension error to reject such checkpoints instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn


LABELS = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
AGENTS = ["Lammy", "Anneena", "Jade", "Penguin"]
OBJECTS = ["Hairbrush", "Sneakers", "Glasses", "Key"]
LOCATIONS = ["Mars", "Mercury", "Venus", "Moon"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")
NEW_CHARACTER_NAMES = {
    "Lammy": "Orion",
    "Anneena": "Bianca",
    "Jade": "Cyrus",
    "Penguin": "Della",
}
CONDITION_ORDER = [
    "standard",
    "longer_stories",
    "withheld_pickup_triples",
    "new_character_names",
    "distractors",
    "paraphrases",
]
CONDITION_TITLES = {
    "standard": "Standard stories",
    "longer_stories": "Longer stories",
    "withheld_pickup_triples": "Withheld pickup triples",
    "new_character_names": "New-name OOV stress test",
    "distractors": "Controlled distractor event",
    "paraphrases": "Paraphrases",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/week6_full_dataset"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("results/tiny_transformer_training"),
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_best.pt",
        choices=("checkpoint_best.pt", "checkpoint_final.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generalization_evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--position-extension",
        choices=("repeat_last", "error"),
        default="repeat_last",
    )
    parser.add_argument(
        "--max-per-condition",
        type=int,
        default=None,
        help="Optional balanced smoke-test size; must be a multiple of five.",
    )
    parser.add_argument(
        "--max-baseline-train",
        type=int,
        default=None,
        help="Optional balanced subset used to fit baselines.",
    )
    parser.add_argument("--bow-max-iterations", type=int, default=2_000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for name in ("max_per_condition", "max_baseline_train"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or value % len(LABELS) != 0):
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be a positive multiple of five")


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
            f"Missing {path}. Generate it with "
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
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no examples")
    return records


def balanced_limit(
    records: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    quota = limit // len(LABELS)
    selected = {label: [] for label in LABELS}
    for record in records:
        answer = record["answer"]
        if len(selected[answer]) < quota:
            selected[answer].append(record)
        if all(len(items) == quota for items in selected.values()):
            break
    if any(len(items) < quota for items in selected.values()):
        raise ValueError("A requested evaluation subset cannot be class-balanced")
    return [selected[label][index] for index in range(quota) for label in LABELS]


def rename_characters(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for original in records:
        record = deepcopy(original)
        text = record["story_text"]
        for old_name, new_name in NEW_CHARACTER_NAMES.items():
            text = re.sub(rf"\b{re.escape(old_name)}\b", new_name, text)
        record["story_text"] = text
        record["evaluation_note"] = (
            "Names are replaced in the input; the target class remains the "
            "corresponding agent slot from the original world."
        )
        renamed.append(record)
    return renamed


def pickup_phrase(objects: list[str]) -> str:
    named = [f"the {object_name.lower()}" for object_name in objects]
    if len(named) == 1:
        return named[0]
    if len(named) == 2:
        return f"{named[0]} and {named[1]}"
    return ", ".join(named[:-1]) + f", and {named[-1]}"


def add_controlled_distractor(record: dict[str, Any]) -> dict[str, Any]:
    """Append one valid move that cannot change the queried object's carrier."""

    changed = deepcopy(record)
    state = changed["final_state"]
    queried_object = changed["question"]["object"]
    queried_value = state["object_states"][queried_object]
    queried_carrier = queried_value if queried_value in AGENTS else None

    choice: tuple[str, str] | None = None
    for agent in AGENTS:
        if agent == queried_carrier:
            continue
        for destination in LOCATIONS:
            if destination == state["agent_locations"][agent]:
                continue
            if queried_carrier is None and destination == queried_value:
                continue
            choice = (agent, destination)
            break
        if choice is not None:
            break
    if choice is None:
        raise ValueError("Could not construct an answer-invariant distractor move")

    agent, destination = choice
    pickups = [
        object_name
        for object_name in OBJECTS
        if state["object_states"][object_name] == destination
    ]
    if queried_object in pickups:
        raise AssertionError("Distractor construction changed the queried carrier")
    state["agent_locations"][agent] = destination
    for object_name in pickups:
        state["object_states"][object_name] = agent

    event_id = max(event["event_id"] for event in changed["events"]) + 1
    event = {
        "action": "move",
        "agent": agent,
        "destination": destination,
        "automatic_pickups": pickups,
        "event_id": event_id,
    }
    changed["events"].append(event)
    changed["metrics"]["story_length"] += 1
    sentence = f"{event_id}. {agent} travels to {destination}"
    if pickups:
        sentence += f" and automatically picks up {pickup_phrase(pickups)}"
    sentence += "."
    prefix, question = changed["story_text"].rsplit("\n\nQuestion:", 1)
    changed["story_text"] = prefix + "\n" + sentence + "\n\nQuestion:" + question
    changed["evaluation_note"] = (
        "One simulator-valid move was appended; it provably leaves the queried "
        "object's carrier unchanged."
    )
    return changed


def build_conditions(
    data_dir: Path, maximum: int | None
) -> dict[str, list[dict[str, Any]]]:
    standard = load_jsonl(data_dir / "test_standard.jsonl")
    conditions = {
        "standard": standard,
        "longer_stories": load_jsonl(data_dir / "test_long.jsonl"),
        "withheld_pickup_triples": load_jsonl(data_dir / "test_withheld.jsonl"),
        "new_character_names": rename_characters(standard),
        "distractors": [add_controlled_distractor(record) for record in standard],
        "paraphrases": load_jsonl(data_dir / "test_paraphrase.jsonl"),
    }
    return {
        name: balanced_limit(records, maximum)
        for name, records in conditions.items()
    }


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


def checkpoint_paths(root: Path, filename: str) -> list[Path]:
    paths = sorted(root.glob(f"seed_*/{filename}"))
    if len(paths) < 3:
        raise FileNotFoundError(
            f"Expected at least three {filename} files below {root}; found {len(paths)}"
        )
    return paths


def maximum_condition_length(
    conditions: dict[str, list[dict[str, Any]]]
) -> int:
    return max(
        len(tokenize(record["story_text"])) + 1
        for records in conditions.values()
        for record in records
    )


def load_transformer(
    path: Path,
    required_capacity: int,
    extension_policy: str,
    device: torch.device,
) -> tuple[TinyTransformer, dict[str, int], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    settings = checkpoint["settings"]
    vocabulary = checkpoint["vocabulary"]
    if checkpoint["labels"] != LABELS:
        raise ValueError(f"{path} has an incompatible label order")
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    state = checkpoint["model_state"]
    learned_positions = state["position_embedding.weight"]
    learned_capacity = learned_positions.shape[0]
    evaluation_capacity = max(learned_capacity, required_capacity)
    extended_positions = evaluation_capacity > learned_capacity
    if extended_positions and extension_policy == "error":
        raise ValueError(
            f"{path} learned {learned_capacity} positions but evaluation needs "
            f"{evaluation_capacity}. Retrain with a larger position capacity or "
            "use --position-extension repeat_last."
        )
    if extended_positions:
        replacement = learned_positions.new_empty(
            evaluation_capacity, learned_positions.shape[1]
        )
        replacement[:learned_capacity] = learned_positions
        replacement[learned_capacity:] = learned_positions[-1]
        state = dict(state)
        state["position_embedding.weight"] = replacement

    model = TinyTransformer(
        vocabulary_size=len(vocabulary),
        position_capacity=evaluation_capacity,
        pad_index=token_to_index["<pad>"],
        d_model=int(settings["d_model"]),
        heads=int(settings["heads"]),
        layers=int(settings["layers"]),
        ff_dim=int(settings["feed_forward_dimension"]),
        dropout=float(settings["dropout"]),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    metadata = {
        "seed": int(checkpoint["seed"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "learned_position_capacity": learned_capacity,
        "evaluation_position_capacity": evaluation_capacity,
        "position_embeddings_extended": extended_positions,
    }
    return model, token_to_index, metadata


def encode_condition(
    records: list[dict[str, Any]], token_to_index: dict[str, int]
) -> tuple[Tensor, Tensor, float]:
    pad_index = token_to_index["<pad>"]
    unknown_index = token_to_index["<unk>"]
    class_to_index = {label: index for index, label in enumerate(LABELS)}
    encoded: list[list[int]] = []
    unknown_count = 0
    ordinary_count = 0
    for record in records:
        tokens = tokenize(record["story_text"])
        ordinary_count += len(tokens)
        unknown_count += sum(token not in token_to_index for token in tokens)
        encoded.append(
            [token_to_index["<cls>"]]
            + [token_to_index.get(token, unknown_index) for token in tokens]
        )
    maximum_length = max(map(len, encoded))
    inputs = torch.full(
        (len(records), maximum_length), pad_index, dtype=torch.long
    )
    for row, token_ids in enumerate(encoded):
        inputs[row, : len(token_ids)] = torch.tensor(token_ids)
    targets = torch.tensor(
        [class_to_index[record["answer"]] for record in records], dtype=torch.long
    )
    unknown_rate = unknown_count / ordinary_count if ordinary_count else 0.0
    return inputs, targets, unknown_rate


@torch.no_grad()
def transformer_predictions(
    model: TinyTransformer,
    inputs: Tensor,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    predictions: list[str] = []
    for offset in range(0, len(inputs), batch_size):
        logits = model(inputs[offset : offset + batch_size].to(device))
        predictions.extend(LABELS[index] for index in logits.argmax(dim=-1).cpu())
    return predictions


def score_predictions(
    records: list[dict[str, Any]], predictions: list[str]
) -> dict[str, Any]:
    truths = [record["answer"] for record in records]
    if len(truths) != len(predictions):
        raise ValueError("Prediction count does not match example count")
    matrix = {truth: {guess: 0 for guess in LABELS} for truth in LABELS}
    correct = 0
    for truth, guess in zip(truths, predictions):
        matrix[truth][guess] += 1
        correct += int(truth == guess)
    return {
        "examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "confusion_matrix": matrix,
        "prediction_counts": dict(Counter(predictions)),
    }


def majority_label(records: list[dict[str, Any]]) -> str:
    counts = Counter(record["answer"] for record in records)
    largest = max(counts.values())
    return next(label for label in LABELS if counts[label] == largest)


def fit_bow(
    records: list[dict[str, Any]], maximum_iterations: int
) -> tuple[CountVectorizer, LogisticRegression]:
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    features = vectorizer.fit_transform(record["story_text"] for record in records)
    model = LogisticRegression(
        C=1.0,
        max_iter=maximum_iterations,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(features, [record["answer"] for record in records])
    return vectorizer, model


def add_result(
    detailed: list[dict[str, Any]],
    model_name: str,
    model_family: str,
    seed: int | None,
    condition: str,
    score: dict[str, Any],
    unknown_rate: float | None = None,
) -> None:
    detailed.append(
        {
            "model": model_name,
            "model_family": model_family,
            "seed": seed,
            "condition": condition,
            "condition_title": CONDITION_TITLES[condition],
            "examples": score["examples"],
            "correct": score["correct"],
            "accuracy": score["accuracy"],
            "unknown_token_rate": unknown_rate,
            "confusion_matrix": score["confusion_matrix"],
            "prediction_counts": score["prediction_counts"],
        }
    )


def build_wide_table(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for result in detailed:
        model_name = result["model"]
        rows.setdefault(model_name, {"model": model_name})
        rows[model_name][result["condition"]] = result["accuracy"]

    transformer_rows = [
        row for row in detailed if row["model_family"] == "transformer"
    ]
    mean_row: dict[str, Any] = {"model": "tiny_transformer_mean"}
    for condition in CONDITION_ORDER:
        values = [
            row["accuracy"]
            for row in transformer_rows
            if row["condition"] == condition
        ]
        mean_row[condition] = statistics.mean(values)
        mean_row[f"{condition}_std"] = statistics.pstdev(values)
    ordered_names = ["majority_baseline", "bag_of_words"] + sorted(
        name for name in rows if name.startswith("tiny_transformer_seed_")
    )
    output = [rows[name] for name in ordered_names if name in rows]
    output.append(mean_row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key not in ("confusion_matrix", "prediction_counts"):
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, table: list[dict[str, Any]]) -> None:
    headers = ["Model"] + [CONDITION_TITLES[name] for name in CONDITION_ORDER]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(CONDITION_ORDER)) + " |",
    ]
    for row in table:
        cells = [row["model"]] + [
            f"{row.get(condition, float('nan')):.2%}" for condition in CONDITION_ORDER
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_records = balanced_limit(
        load_jsonl(args.data_dir / "train.jsonl"), args.max_baseline_train
    )
    conditions = build_conditions(args.data_dir, args.max_per_condition)
    required_capacity = maximum_condition_length(conditions)
    paths = checkpoint_paths(args.checkpoint_root, args.checkpoint_name)

    detailed: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    print("Reusable generalization evaluation")
    print(f"Device: {device} | checkpoints: {len(paths)}")
    for path in paths:
        model, token_to_index, metadata = load_transformer(
            path, required_capacity, args.position_extension, device
        )
        metadata["path"] = str(path.resolve())
        checkpoint_metadata.append(metadata)
        model_name = f"tiny_transformer_seed_{metadata['seed']}"
        for condition in CONDITION_ORDER:
            records = conditions[condition]
            inputs, _, unknown_rate = encode_condition(records, token_to_index)
            predictions = transformer_predictions(
                model, inputs, args.batch_size, device
            )
            score = score_predictions(records, predictions)
            add_result(
                detailed,
                model_name,
                "transformer",
                metadata["seed"],
                condition,
                score,
                unknown_rate,
            )
            print(
                f"{model_name:25s} | {condition:35s} | {score['accuracy']:.2%}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    majority = majority_label(training_records)
    vectorizer, bow_model = fit_bow(training_records, args.bow_max_iterations)
    for condition in CONDITION_ORDER:
        records = conditions[condition]
        majority_score = score_predictions(records, [majority] * len(records))
        add_result(
            detailed,
            "majority_baseline",
            "baseline",
            None,
            condition,
            majority_score,
        )
        bow_predictions = bow_model.predict(
            vectorizer.transform(record["story_text"] for record in records)
        ).tolist()
        bow_score = score_predictions(records, bow_predictions)
        add_result(
            detailed,
            "bag_of_words",
            "baseline",
            None,
            condition,
            bow_score,
        )

    table = build_wide_table(detailed)
    metadata = {
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_metadata": checkpoint_metadata,
        "position_extension_policy": args.position_extension,
        "baseline_training_examples": len(training_records),
        "condition_examples": {
            name: len(records) for name, records in conditions.items()
        },
        "condition_definitions": {
            "standard": "Canonical 1-6 event test split.",
            "longer_stories": "Canonical 7-10 event test split.",
            "withheld_pickup_triples": (
                "Generator-withheld agent-object-destination pickup triples. "
                "The training set contains the corresponding person-object pairs "
                "at other destinations, so this is not a true unseen-pair test."
            ),
            "new_character_names": (
                f"Standard stories with {NEW_CHARACTER_NAMES}; the fixed output "
                "classes retain their original agent-slot meanings. All genuinely "
                "new word tokens map to <unk> for the transformer."
            ),
            "distractors": (
                "A paired transformation of every standard story with one extra "
                "simulator-valid event proven not to change the queried carrier."
            ),
            "paraphrases": "Dedicated paraphrase test split.",
        },
        "majority_label": majority,
        "bag_of_words_vocabulary_size": len(vectorizer.vocabulary_),
    }
    with (args.output_dir / "complete_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"metadata": metadata, "comparison_table": table, "details": detailed},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    write_csv(args.output_dir / "comparison_table.csv", table)
    write_csv(args.output_dir / "detailed_results.csv", detailed)
    write_markdown(args.output_dir / "comparison_table.md", table)

    print("\nAutomatic comparison table")
    print((args.output_dir / "comparison_table.md").read_text(encoding="utf-8"))
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

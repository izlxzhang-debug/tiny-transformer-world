#!/usr/bin/env python3
"""Train transformer ablations and categorize their incorrect answers.

Full experiment, from the repository root:

    python3 week11/error_analysis_and_ablations.py

Fast five-model pipeline check:

    python3 week11/error_analysis_and_ablations.py \
        --device cpu --epochs 1 --d-model 32 --ff-dim 64 \
        --train-limit 100 --validation-limit 100 --test-limit 20 \
        --fewer-examples 20 --output-dir results/week11_smoke

The program trains a reference model and four controlled ablations:

1. no positional information;
2. one transformer layer instead of two;
3. irrelevant training events removed;
4. fewer training examples.

It then evaluates paired standard, distractor, renamed-character, and
paraphrased stories plus a longer-story split. Incorrect answers receive
reproducible, nonexclusive diagnostic labels. These labels identify behavior
consistent with a failure mode; they do not prove the model's internal cause.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


LABELS = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
AGENTS = ["Lammy", "Anneena", "Jade", "Penguin"]
OBJECTS = ["Hairbrush", "Sneakers", "Glasses", "Key"]
LOCATIONS = ["Mars", "Mercury", "Venus", "Moon"]
SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")
NEW_NAMES = {
    "Lammy": "Orion",
    "Anneena": "Bianca",
    "Jade": "Cyrus",
    "Penguin": "Della",
}
CONDITIONS = [
    "standard",
    "longer_chains",
    "distractors",
    "new_character_names",
    "paraphrased_wording",
]
ERROR_CATEGORIES = [
    "most_recent_room_shortcut",
    "failure_to_move_carried_objects",
    "confusion_caused_by_distractors",
    "failure_on_longer_chains",
    "dependence_on_familiar_names",
    "dependence_on_exact_wording",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/week6_full_dataset")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/week11_ablations")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--fewer-examples",
        type=int,
        default=2_000,
        help="Training size for the fewer-examples ablation; multiple of 20.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional balanced reference-training limit; multiple of 20.",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
        help="Optional balanced validation limit; multiple of 20.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional balanced examples per test condition; multiple of five.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.d_model <= 0 or args.heads <= 0 or args.d_model % args.heads:
        raise ValueError("d-model must be positive and divisible by heads")
    if args.ff_dim <= 0 or not 0.0 <= args.dropout < 1.0:
        raise ValueError("invalid feed-forward dimension or dropout")
    for name in ("train_limit", "validation_limit", "fewer_examples"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or value % 20):
            raise ValueError(f"--{name.replace('_', '-')} must be a positive multiple of 20")
    if args.test_limit is not None and (
        args.test_limit <= 0 or args.test_limit % len(LABELS)
    ):
        raise ValueError("--test-limit must be a positive multiple of five")
    if args.train_limit is not None and args.fewer_examples >= args.train_limit:
        raise ValueError("--fewer-examples must be smaller than --train-limit")


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


def balanced_answer_limit(
    records: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    quota = limit // len(LABELS)
    selected = {label: [] for label in LABELS}
    for record in records:
        label = record["answer"]
        if len(selected[label]) < quota:
            selected[label].append(record)
        if all(len(items) == quota for items in selected.values()):
            break
    if any(len(items) < quota for items in selected.values()):
        raise ValueError("Could not create a class-balanced subset")
    return [selected[label][i] for i in range(quota) for label in LABELS]


def balanced_cell_limit(
    records: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    quota = limit // (len(OBJECTS) * len(LABELS))
    cells = {
        (object_name, label): [] for object_name in OBJECTS for label in LABELS
    }
    for record in records:
        cell = (record["question"]["object"], record["answer"])
        if len(cells[cell]) < quota:
            cells[cell].append(record)
        if all(len(items) == quota for items in cells.values()):
            break
    if any(len(items) < quota for items in cells.values()):
        raise ValueError("Could not create an object-answer-balanced subset")
    return [
        cells[(object_name, label)][i]
        for i in range(quota)
        for object_name in OBJECTS
        for label in LABELS
    ]


def rename_story(record: dict[str, Any]) -> dict[str, Any]:
    changed = deepcopy(record)
    text = changed["story_text"]
    for old, new in NEW_NAMES.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    changed["story_text"] = text
    return changed


def paraphrase_story(record: dict[str, Any]) -> dict[str, Any]:
    changed = deepcopy(record)
    text = changed["story_text"]
    text = re.sub(
        r"\b(Lammy|Anneena|Jade|Penguin) is on (Mars|Mercury|Venus|Moon)\.",
        r"\1 is located on \2.",
        text,
    )
    text = re.sub(
        r"The (hairbrush|sneakers|glasses|key) is lying on "
        r"(Mars|Mercury|Venus|Moon)\.",
        r"At first, the \1 lies on \2.",
        text,
    )
    text = text.replace(" travels to ", " heads to ")
    text = text.replace(" automatically picks up ", " collects ")
    text = re.sub(
        r"\b(Lammy|Anneena|Jade|Penguin) drops the "
        r"(hairbrush|sneakers|glasses|key) on (Mars|Mercury|Venus|Moon)\.",
        r"\1 sets down the \2 on \3.",
        text,
    )
    changed["story_text"] = text
    return changed


def pickup_phrase(objects: list[str]) -> str:
    names = [f"the {name.lower()}" for name in objects]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def add_distractor(record: dict[str, Any]) -> dict[str, Any]:
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
        if choice:
            break
    if choice is None:
        raise ValueError("Could not construct an answer-invariant distractor")
    agent, destination = choice
    pickups = [
        object_name
        for object_name in OBJECTS
        if state["object_states"][object_name] == destination
    ]
    if queried_object in pickups:
        raise AssertionError("Distractor changes the queried carrier")
    state["agent_locations"][agent] = destination
    for object_name in pickups:
        state["object_states"][object_name] = agent
    event_id = max(event["event_id"] for event in changed["events"]) + 1
    changed["events"].append(
        {
            "action": "move",
            "agent": agent,
            "destination": destination,
            "automatic_pickups": pickups,
            "event_id": event_id,
        }
    )
    changed["metrics"]["story_length"] += 1
    sentence = f"{event_id}. {agent} travels to {destination}"
    if pickups:
        sentence += f" and automatically picks up {pickup_phrase(pickups)}"
    sentence += "."
    prefix, question = changed["story_text"].rsplit("\n\nQuestion:", 1)
    changed["story_text"] = prefix + "\n" + sentence + "\n\nQuestion:" + question
    return changed


def remove_distractor_events(record: dict[str, Any]) -> dict[str, Any]:
    """Remove event sentences that never change the queried object's state."""

    changed = deepcopy(record)
    evidence = set(changed.get("evidence_event_ids", []))
    prefix, remainder = changed["story_text"].split("\n\nEvents:\n", 1)
    event_text, question = remainder.rsplit("\n\nQuestion:", 1)
    kept_lines: list[str] = []
    for line in event_text.splitlines():
        match = re.match(r"(\d+)\.", line)
        if match and int(match.group(1)) in evidence:
            kept_lines.append(line)
    changed["story_text"] = (
        prefix + "\n\nEvents:\n" + "\n".join(kept_lines) + "\n\nQuestion:" + question
    )
    return changed


def build_vocabulary(records: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    counts = Counter(
        token for record in records for token in tokenize(record["story_text"])
    )
    vocabulary = SPECIAL_TOKENS + sorted(counts)
    return {token: index for index, token in enumerate(vocabulary)}, vocabulary


def encode_records(
    records: list[dict[str, Any]], token_to_index: dict[str, int]
) -> tuple[Tensor, Tensor]:
    pad = token_to_index["<pad>"]
    unknown = token_to_index["<unk>"]
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    encoded = [
        [token_to_index["<cls>"]]
        + [token_to_index.get(token, unknown) for token in tokenize(r["story_text"])]
        for r in records
    ]
    maximum_length = max(map(len, encoded))
    inputs = torch.full((len(records), maximum_length), pad, dtype=torch.long)
    for row, token_ids in enumerate(encoded):
        inputs[row, : len(token_ids)] = torch.tensor(token_ids)
    targets = torch.tensor(
        [label_to_index[record["answer"]] for record in records], dtype=torch.long
    )
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

        def split_heads(value: Tensor) -> Tensor:
            return value.view(
                batch_size, sequence_length, self.heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(self.q_projection(x))
        keys = split_heads(self.k_projection(x))
        values = split_heads(self.v_projection(x))
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attended = self.dropout(torch.softmax(scores, dim=-1)) @ values
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
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        use_positions: bool,
    ) -> None:
        super().__init__()
        self.pad_index = 0
        self.use_positions = use_positions
        self.token_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=0)
        self.position_embedding = (
            nn.Embedding(position_capacity, d_model) if use_positions else None
        )
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
        x = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=token_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, sequence_length)
            x = x + self.position_embedding(positions)
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


def train_epoch(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float, float]:
    model.train()
    order = torch.randperm(len(inputs), generator=generator)
    total_loss = 0.0
    correct = 0
    for offset in range(0, len(inputs), batch_size):
        indices = order[offset : offset + batch_size]
        batch_inputs = inputs[indices].to(device)
        batch_targets = targets[indices].to(device)
        logits = model(batch_inputs)
        loss = loss_function(logits, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(indices)
        correct += logits.argmax(dim=-1).eq(batch_targets).sum().item()
    return total_loss / len(inputs), correct / len(inputs)


@torch.no_grad()
def predict(
    model: nn.Module,
    inputs: Tensor,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    model.eval()
    output: list[str] = []
    for offset in range(0, len(inputs), batch_size):
        logits = model(inputs[offset : offset + batch_size].to(device))
        output.extend(LABELS[index] for index in logits.argmax(dim=-1).cpu())
    return output


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    loss_function: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for offset in range(0, len(inputs), batch_size):
        batch_inputs = inputs[offset : offset + batch_size].to(device)
        batch_targets = targets[offset : offset + batch_size].to(device)
        logits = model(batch_inputs)
        loss = loss_function(logits, batch_targets)
        total_loss += loss.item() * len(batch_inputs)
        correct += logits.argmax(dim=-1).eq(batch_targets).sum().item()
    return total_loss / len(inputs), correct / len(inputs)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_configuration(
    name: str,
    config: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_tensors: tuple[Tensor, Tensor],
    token_to_index: dict[str, int],
    vocabulary: list[str],
    position_capacity: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TinyTransformer, list[dict[str, Any]], float]:
    set_seed(args.seed)
    directory = args.output_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    training_inputs, training_targets = encode_records(train_records, token_to_index)
    validation_inputs, validation_targets = validation_tensors
    model = TinyTransformer(
        vocabulary_size=len(vocabulary),
        position_capacity=position_capacity,
        d_model=args.d_model,
        heads=args.heads,
        layers=config["layers"],
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_positions=config["use_positions"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            training_inputs,
            training_targets,
            optimizer,
            loss_function,
            args.batch_size,
            device,
            generator,
        )
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_inputs,
            validation_targets,
            loss_function,
            args.batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "training_loss": train_loss,
            "training_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"{name:28s} | epoch {epoch:02d}/{args.epochs} "
            f"| train {train_accuracy:.2%} | validation {validation_accuracy:.2%}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "configuration": config,
                    "epoch": epoch,
                    "seed": args.seed,
                    "vocabulary": vocabulary,
                },
                directory / "checkpoint_best.pt",
            )
    duration = time.perf_counter() - started
    torch.save(
        {
            "model_state": model.state_dict(),
            "configuration": config,
            "epoch": args.epochs,
            "seed": args.seed,
            "vocabulary": vocabulary,
        },
        directory / "checkpoint_final.pt",
    )
    best = torch.load(directory / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best["model_state"])
    settings = {
        **config,
        "seed": args.seed,
        "training_examples": len(train_records),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "heads": args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "position_capacity": position_capacity,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_updates": math.ceil(len(train_records) / args.batch_size)
        * args.epochs,
        "duration_seconds": duration,
        "best_epoch": best["epoch"],
    }
    write_json(directory / "settings.json", settings)
    write_json(directory / "history.json", history)
    write_csv(directory / "history.csv", history)
    return model, history, duration


def most_recent_arrival(record: dict[str, Any]) -> str | None:
    location = record.get("answer_details", {}).get("effective_location")
    if location not in LOCATIONS:
        return None
    for event in reversed(record["events"]):
        if event["action"] == "move" and event["destination"] == location:
            return event["agent"]
    return None


def failure_to_move_counterfactual(record: dict[str, Any]) -> str:
    """Replay with the bug that a carried queried object stays at the old room."""

    queried_object = record["question"]["object"]
    locations = dict(record["initial_state"]["agent_locations"])
    holder_or_location = record["initial_state"]["object_states"][queried_object]
    for event in record["events"]:
        agent = event["agent"]
        if event["action"] == "move":
            origin = locations[agent]
            if holder_or_location == agent:
                holder_or_location = origin
            locations[agent] = event["destination"]
            if holder_or_location == event["destination"]:
                holder_or_location = agent
        elif event["action"] == "drop" and event["object"] == queried_object:
            if holder_or_location == agent:
                holder_or_location = locations[agent]
    return holder_or_location if holder_or_location in AGENTS else "Nobody"


def categorize_error(
    record: dict[str, Any],
    prediction: str,
    condition: str,
    standard_prediction: str | None,
) -> list[str]:
    categories: list[str] = []
    if prediction == most_recent_arrival(record):
        categories.append("most_recent_room_shortcut")
    buggy_answer = failure_to_move_counterfactual(record)
    if buggy_answer != record["answer"] and prediction == buggy_answer:
        categories.append("failure_to_move_carried_objects")
    if condition == "distractors" and standard_prediction == record["answer"]:
        categories.append("confusion_caused_by_distractors")
    if condition == "longer_chains":
        categories.append("failure_on_longer_chains")
    if condition == "new_character_names" and standard_prediction == record["answer"]:
        categories.append("dependence_on_familiar_names")
    if condition == "paraphrased_wording" and standard_prediction == record["answer"]:
        categories.append("dependence_on_exact_wording")
    return categories or ["other"]


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_training = balanced_cell_limit(
        load_jsonl(args.data_dir / "train.jsonl"), args.train_limit
    )
    validation_records = balanced_cell_limit(
        load_jsonl(args.data_dir / "validation.jsonl"), args.validation_limit
    )
    standard = balanced_answer_limit(
        load_jsonl(args.data_dir / "test_standard.jsonl"), args.test_limit
    )
    longer = balanced_answer_limit(
        load_jsonl(args.data_dir / "test_long.jsonl"), args.test_limit
    )
    conditions = {
        "standard": standard,
        "longer_chains": longer,
        "distractors": [add_distractor(record) for record in standard],
        "new_character_names": [rename_story(record) for record in standard],
        "paraphrased_wording": [paraphrase_story(record) for record in standard],
    }
    token_to_index, vocabulary = build_vocabulary(original_training)
    validation_tensors = encode_records(validation_records, token_to_index)
    condition_tensors = {
        name: encode_records(records, token_to_index)
        for name, records in conditions.items()
    }
    position_capacity = max(
        inputs.shape[1]
        for inputs, _ in [validation_tensors, *condition_tensors.values()]
    )

    fewer_count = min(args.fewer_examples, len(original_training))
    if fewer_count == len(original_training):
        raise ValueError("fewer-examples ablation is not smaller than reference data")
    fewer_training = balanced_cell_limit(original_training, fewer_count)
    no_distractor_training = [
        remove_distractor_events(record) for record in original_training
    ]
    configurations = {
        "reference": {
            "use_positions": True,
            "layers": 2,
            "training_variant": "original",
        },
        "no_positional_information": {
            "use_positions": False,
            "layers": 2,
            "training_variant": "original",
        },
        "one_layer": {
            "use_positions": True,
            "layers": 1,
            "training_variant": "original",
        },
        "remove_training_distractors": {
            "use_positions": True,
            "layers": 2,
            "training_variant": "irrelevant_events_removed",
        },
        "fewer_training_examples": {
            "use_positions": True,
            "layers": 2,
            "training_variant": "fewer_examples",
        },
    }
    training_sets = {
        "reference": original_training,
        "no_positional_information": original_training,
        "one_layer": original_training,
        "remove_training_distractors": no_distractor_training,
        "fewer_training_examples": fewer_training,
    }

    ablation_rows: list[dict[str, Any]] = []
    prediction_lookup: dict[tuple[str, str, str], str] = {}
    all_models: dict[str, TinyTransformer] = {}
    for name, config in configurations.items():
        model, history, duration = train_configuration(
            name,
            config,
            training_sets[name],
            validation_tensors,
            token_to_index,
            vocabulary,
            position_capacity,
            args,
            device,
        )
        all_models[name] = model
        row: dict[str, Any] = {
            "model": name,
            "positions": config["use_positions"],
            "layers": config["layers"],
            "training_variant": config["training_variant"],
            "training_examples": len(training_sets[name]),
            "best_validation_accuracy": max(
                item["validation_accuracy"] for item in history
            ),
            "duration_seconds": duration,
        }
        for condition, records in conditions.items():
            inputs, _ = condition_tensors[condition]
            predictions = predict(model, inputs, args.batch_size, device)
            accuracy = sum(
                prediction == record["answer"]
                for prediction, record in zip(predictions, records)
            ) / len(records)
            row[f"{condition}_accuracy"] = accuracy
            for record, prediction in zip(records, predictions):
                prediction_lookup[(name, condition, record["story_id"])] = prediction
        ablation_rows.append(row)
        print(f"completed evaluation: {name}")

    incorrect_rows: list[dict[str, Any]] = []
    category_counts: dict[str, Counter[str]] = {
        name: Counter() for name in configurations
    }
    error_totals = Counter()
    for model_name in configurations:
        for condition, records in conditions.items():
            for record in records:
                story_id = record["story_id"]
                prediction = prediction_lookup[(model_name, condition, story_id)]
                if prediction == record["answer"]:
                    continue
                error_totals[model_name] += 1
                standard_prediction = prediction_lookup.get(
                    (model_name, "standard", story_id)
                )
                categories = categorize_error(
                    record, prediction, condition, standard_prediction
                )
                for category in categories:
                    category_counts[model_name][category] += 1
                incorrect_rows.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "story_id": story_id,
                        "truth": record["answer"],
                        "prediction": prediction,
                        "story_length": record["metrics"]["story_length"],
                        "reasoning_depth": record["metrics"]["reasoning_depth"],
                        "categories": ";".join(categories),
                        "primary_category": categories[0],
                    }
                )

    category_rows: list[dict[str, Any]] = []
    for model_name in configurations:
        total = error_totals[model_name]
        for category in ERROR_CATEGORIES:
            count = category_counts[model_name][category]
            category_rows.append(
                {
                    "model": model_name,
                    "category": category,
                    "count": count,
                    "fraction_of_incorrect_answers": count / total if total else 0.0,
                    "total_incorrect_answers": total,
                }
            )

    metadata = {
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "vocabulary_size": len(vocabulary),
        "position_capacity": position_capacity,
        "error_category_note": (
            "Categories are nonexclusive behavioral diagnostics, not proof of "
            "the model's internal causal mechanism."
        ),
        "shared_vocabulary_note": (
            "Every ablation uses the vocabulary learned from the reference "
            "training set so embedding dimensions remain controlled. The "
            "fewer-examples ablation therefore has unlabeled vocabulary exposure."
        ),
        "category_definitions": {
            "most_recent_room_shortcut": (
                "The wrong predicted agent is the most recent agent to move to "
                "the queried object's final physical location."
            ),
            "failure_to_move_carried_objects": (
                "The wrong prediction exactly matches a counterfactual simulator "
                "that incorrectly leaves the queried object in the carrier's old "
                "room whenever that carrier moves."
            ),
            "confusion_caused_by_distractors": (
                "The paired standard story was correct, but adding one valid, "
                "answer-invariant event made the answer wrong."
            ),
            "failure_on_longer_chains": "An incorrect answer on a 7-10 event story.",
            "dependence_on_familiar_names": (
                "The paired standard story was correct, but replacing all four "
                "agent names with unseen names made the answer wrong."
            ),
            "dependence_on_exact_wording": (
                "The paired standard story was correct, but deterministic wording "
                "changes made the answer wrong."
            ),
            "other": "No listed behavioral signature applied.",
        },
    }
    write_json(
        args.output_dir / "complete_results.json",
        {
            "metadata": metadata,
            "ablation_table": ablation_rows,
            "error_category_table": category_rows,
            "incorrect_answers": incorrect_rows,
        },
    )
    write_csv(args.output_dir / "ablation_table.csv", ablation_rows)
    write_csv(args.output_dir / "error_categories.csv", category_rows)
    write_csv(args.output_dir / "incorrect_answers.csv", incorrect_rows)

    print("\nAblation results")
    for row in ablation_rows:
        print(
            f"{row['model']:28s} | standard {row['standard_accuracy']:.2%} "
            f"| long {row['longer_chains_accuracy']:.2%} "
            f"| distractor {row['distractors_accuracy']:.2%} "
            f"| names {row['new_character_names_accuracy']:.2%} "
            f"| wording {row['paraphrased_wording_accuracy']:.2%}"
        )
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

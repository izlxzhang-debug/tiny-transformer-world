#!/usr/bin/env python3
"""Evaluate a majority-class baseline on the Week 6 world-story dataset.

The model learns only the most common answer in the training split and predicts
that same answer for every example. It never reads the story text.

Run from the repository root:
    python3 scripts/evaluate_majority_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LABEL_ORDER = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
TEST_FILES = {
    "standard": "test_standard.jsonl",
    "long": "test_long.jsonl",
    "paraphrase": "test_paraphrase.jsonl",
    "withheld": "test_withheld.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/week6_full_dataset"),
        help="Directory containing the Week 6 JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/majority_baseline"),
        help="Directory in which evaluation results will be saved.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate the Week 6 dataset first with: "
            "python3 scripts/generate_week6_full_dataset.py"
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "answer" not in record or "metrics" not in record:
                raise ValueError(f"{path}:{line_number} has an invalid record format")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no examples")
    return records


def choose_majority_label(training_records: list[dict[str, Any]]) -> tuple[str, Counter[str]]:
    counts = Counter(record["answer"] for record in training_records)
    largest_count = max(counts.values())
    tied_labels = {label for label, count in counts.items() if count == largest_count}

    # The generated training data is deliberately balanced, so all five labels
    # may tie. Use the declared label order to make that tie deterministic.
    for label in DEFAULT_LABEL_ORDER:
        if label in tied_labels:
            return label, counts
    return sorted(tied_labels)[0], counts


def ordered_labels(records: list[dict[str, Any]], prediction: str) -> list[str]:
    observed = {record["answer"] for record in records} | {prediction}
    labels = [label for label in DEFAULT_LABEL_ORDER if label in observed]
    labels.extend(sorted(observed - set(labels)))
    return labels


def confusion_matrix(
    true_labels: list[str], predicted_labels: list[str], labels: list[str]
) -> list[list[int]]:
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(true_labels, predicted_labels):
        matrix[positions[truth]][positions[prediction]] += 1
    return matrix


def evaluate_split(
    records: list[dict[str, Any]], prediction: str
) -> dict[str, Any]:
    true_labels = [record["answer"] for record in records]
    predicted_labels = [prediction] * len(records)
    correct = [truth == guess for truth, guess in zip(true_labels, predicted_labels)]
    labels = ordered_labels(records, prediction)

    length_totals: Counter[int] = Counter()
    length_correct: Counter[int] = Counter()
    for record, is_correct in zip(records, correct):
        story_length = int(record["metrics"]["story_length"])
        length_totals[story_length] += 1
        length_correct[story_length] += int(is_correct)

    return {
        "examples": len(records),
        "correct": sum(correct),
        "accuracy": sum(correct) / len(records),
        "labels": labels,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels),
        "accuracy_by_story_length": {
            str(length): {
                "examples": length_totals[length],
                "correct": length_correct[length],
                "accuracy": length_correct[length] / length_totals[length],
            }
            for length in sorted(length_totals)
        },
    }


def write_results(
    output_dir: Path,
    majority_label: str,
    training_counts: Counter[str],
    split_results: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "majority_class_baseline",
        "description": "Always predicts the most frequent training answer.",
        "majority_label": majority_label,
        "training_answer_counts": dict(training_counts),
        "test_results": split_results,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with (output_dir / "overall_accuracy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["test_set", "examples", "correct", "accuracy"]
        )
        writer.writeheader()
        for split_name, result in split_results.items():
            writer.writerow(
                {
                    "test_set": split_name,
                    "examples": result["examples"],
                    "correct": result["correct"],
                    "accuracy": result["accuracy"],
                }
            )

    with (output_dir / "accuracy_by_story_length.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["test_set", "story_length", "examples", "correct", "accuracy"],
        )
        writer.writeheader()
        for split_name, result in split_results.items():
            for length, metrics in result["accuracy_by_story_length"].items():
                writer.writerow(
                    {
                        "test_set": split_name,
                        "story_length": length,
                        **metrics,
                    }
                )

    for split_name, result in split_results.items():
        labels = result["labels"]
        matrix = result["confusion_matrix"]
        with (output_dir / f"confusion_matrix_{split_name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\predicted", *labels])
            for label, row in zip(labels, matrix):
                writer.writerow([label, *row])


def main() -> None:
    args = parse_args()
    training_records = load_jsonl(args.data_dir / "train.jsonl")
    majority_label, training_counts = choose_majority_label(training_records)

    split_results: dict[str, dict[str, Any]] = {}
    for split_name, filename in TEST_FILES.items():
        records = load_jsonl(args.data_dir / filename)
        split_results[split_name] = evaluate_split(records, majority_label)

    write_results(args.output_dir, majority_label, training_counts, split_results)

    print("Majority-class baseline")
    print(f"Training answer counts: {dict(training_counts)}")
    print(f"Prediction for every story: {majority_label}")
    for split_name, result in split_results.items():
        print(
            f"{split_name:10s}: {result['accuracy']:.2%} "
            f"({result['correct']}/{result['examples']})"
        )
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

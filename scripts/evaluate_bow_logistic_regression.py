#!/usr/bin/env python3
"""Train and evaluate a bag-of-words logistic-regression baseline.

The input is an unordered count of individual words in each story. The model
therefore tests whether word frequencies alone can predict the answer without
explicitly simulating the world or representing event order.

Run from the repository root:
    python3 scripts/evaluate_bow_logistic_regression.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


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
        default=Path("results/bow_logistic_regression"),
        help="Directory in which evaluation results will be saved.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2_000,
        help="Maximum logistic-regression optimization iterations.",
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
            required = {"story_text", "answer", "metrics"}
            if not required.issubset(record):
                raise ValueError(f"{path}:{line_number} has an invalid record format")
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no examples")
    return records


def ordered_labels(training_records: list[dict[str, Any]]) -> list[str]:
    observed = {record["answer"] for record in training_records}
    labels = [label for label in DEFAULT_LABEL_ORDER if label in observed]
    labels.extend(sorted(observed - set(labels)))
    return labels


def evaluate_split(
    records: list[dict[str, Any]],
    vectorizer: CountVectorizer,
    model: LogisticRegression,
    labels: list[str],
) -> dict[str, Any]:
    texts = [record["story_text"] for record in records]
    true_labels = [record["answer"] for record in records]
    features = vectorizer.transform(texts)
    predictions = model.predict(features).tolist()
    correct = [truth == guess for truth, guess in zip(true_labels, predictions)]

    length_totals: Counter[int] = Counter()
    length_correct: Counter[int] = Counter()
    for record, is_correct in zip(records, correct):
        story_length = int(record["metrics"]["story_length"])
        length_totals[story_length] += 1
        length_correct[story_length] += int(is_correct)

    matrix = confusion_matrix(true_labels, predictions, labels=labels)
    return {
        "examples": len(records),
        "correct": sum(correct),
        "accuracy": sum(correct) / len(records),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "prediction_counts": dict(Counter(predictions)),
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
    vocabulary_size: int,
    training_counts: Counter[str],
    split_results: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "bag_of_words_logistic_regression",
        "description": "Unigram word counts followed by multinomial classification.",
        "training_answer_counts": dict(training_counts),
        "vocabulary_size": vocabulary_size,
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
    training_texts = [record["story_text"] for record in training_records]
    training_labels = [record["answer"] for record in training_records]

    # Unigrams only: no bigrams, sequence model, simulator features, or test fit.
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    training_features = vectorizer.fit_transform(training_texts)
    model = LogisticRegression(
        C=1.0,
        max_iter=args.max_iterations,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(training_features, training_labels)

    labels = ordered_labels(training_records)
    split_results: dict[str, dict[str, Any]] = {}
    for split_name, filename in TEST_FILES.items():
        records = load_jsonl(args.data_dir / filename)
        unknown_labels = {record["answer"] for record in records} - set(labels)
        if unknown_labels:
            raise ValueError(
                f"{filename} contains labels absent from training: {sorted(unknown_labels)}"
            )
        split_results[split_name] = evaluate_split(
            records, vectorizer, model, labels
        )

    training_counts = Counter(training_labels)
    write_results(
        args.output_dir,
        len(vectorizer.vocabulary_),
        training_counts,
        split_results,
    )

    print("Bag-of-words logistic-regression baseline")
    print(f"Training examples: {len(training_records):,}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    for split_name, result in split_results.items():
        print(
            f"{split_name:10s}: {result['accuracy']:.2%} "
            f"({result['correct']}/{result['examples']})"
        )
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

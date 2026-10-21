#!/usr/bin/env python3
"""Compare two simple ways of guessing the carrier, without a transformer.

One system always predicts the most common training answer. The other learns
from unordered word counts using logistic regression. This single file needs
Python and scikit-learn, plus the story dataset; no other project code is used.

Run from the repository root:
    python3 'Training and Testing Transformer/Simple Guesses/simple_guesses.py'

Both systems use only training data to fit their parameters. Results on the
four original test sets are saved together in the adjacent results.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


DEFAULT_LABEL_ORDER = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
TEST_SPLITS = {
    "standard": "test_standard",
    "long": "test_long",
    "paraphrase": "test_paraphrase",
    "withheld": "test_withheld",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=Path("Dataset/Training and Harder Tests/dataset.json"),
        help="The complete JSON story dataset with named splits.")
    parser.add_argument("--data-dir", type=Path, default=None,
        help="Optional folder containing the earlier separate JSONL splits.")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--results-file", type=Path,
        default=Path(__file__).resolve().with_name("results.json"))
    return parser.parse_args()


def iter_dataset_records(path: Path, split: str | None = None):
    """Read a named JSON split or the earlier separate JSONL files."""
    selected_split = split or path.stem
    source = path if path.is_file() else path.parent / "dataset.json"
    if not source.is_file():
        root = next(
            (parent for parent in Path(__file__).resolve().parents
             if (parent / "Dataset").is_dir()), Path.cwd(),
        )
        allowed = {root / "data/week6_full_dataset", root / "Dataset/Training and Harder Tests"}
        source = root / "Dataset/Training and Harder Tests/dataset.json"
        if path.parent.resolve() not in allowed or not source.is_file():
            raise FileNotFoundError(f"Missing dataset: {path}")
    with source.open(encoding="utf-8") as handle:
        if source.suffix == ".json":
            bundle = json.load(handle)
            if selected_split not in bundle["splits"]:
                raise ValueError(f"Unknown dataset split: {selected_split}")
            yield from bundle["splits"][selected_split]
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    records = list(iter_dataset_records(path, split))
    if not records:
        raise ValueError(f"Dataset split {split} is empty")
    for record in records:
        if not {"story_text", "answer", "metrics"}.issubset(record):
            raise ValueError(f"Invalid record in {split}")
    return records


def choose_majority_label(training_records):
    counts = Counter(record["answer"] for record in training_records)
    largest = max(counts.values())
    tied = {label for label, count in counts.items() if count == largest}
    for label in DEFAULT_LABEL_ORDER:
        if label in tied:
            return label, counts
    return sorted(tied)[0], counts


def evaluate_predictions(records, predictions, labels):
    if len(records) != len(predictions):
        raise ValueError("There must be one prediction for each story")
    truth = [record["answer"] for record in records]
    correct = [actual == predicted for actual, predicted in zip(truth, predictions)]
    totals, successes = Counter(), Counter()
    for record, matched in zip(records, correct):
        length = int(record["metrics"]["story_length"])
        totals[length] += 1
        successes[length] += int(matched)
    return {
        "examples": len(records),
        "correct": sum(correct),
        "accuracy": sum(correct) / len(records),
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
        "accuracy_by_story_length": {
            str(length): {
                "examples": totals[length],
                "correct": successes[length],
                "accuracy": successes[length] / totals[length],
            }
            for length in sorted(totals)
        },
    }


def run_baselines(training_records, test_sets, max_iterations=2000):
    majority, counts = choose_majority_label(training_records)
    observed = set(counts)
    labels = [label for label in DEFAULT_LABEL_ORDER if label in observed]
    labels += sorted(observed - set(labels))
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    features = vectorizer.fit_transform([r["story_text"] for r in training_records])
    classifier = LogisticRegression(
        C=1.0, max_iter=max_iterations, solver="lbfgs", random_state=42,
    )
    classifier.fit(features, [r["answer"] for r in training_records])

    majority_results, word_results = {}, {}
    for name, records in test_sets.items():
        unknown = {r["answer"] for r in records} - observed
        if unknown:
            raise ValueError(f"{name} has answers absent from training: {sorted(unknown)}")
        majority_results[name] = evaluate_predictions(records, [majority] * len(records), labels)
        predictions = classifier.predict(
            vectorizer.transform([r["story_text"] for r in records])
        ).tolist()
        word_results[name] = evaluate_predictions(records, predictions, labels)
        word_results[name]["prediction_counts"] = dict(Counter(predictions))

    return {
        "most_common_answer": {
            "model": "majority_class_baseline",
            "description": "Always predicts the most frequent training answer.",
            "majority_label": majority,
            "training_answer_counts": dict(counts),
            "test_results": majority_results,
        },
        "word_counts": {
            "model": "bag_of_words_logistic_regression",
            "description": "Unigram word counts followed by multinomial classification.",
            "training_answer_counts": dict(counts),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "test_results": word_results,
        },
    }


def main():
    args = parse_args()
    def read(split):
        path = args.data_dir / f"{split}.jsonl" if args.data_dir else args.data
        return load_split(path, split)
    training_records = read("train")
    test_sets = {name: read(split) for name, split in TEST_SPLITS.items()}
    result = {
        "experiment": "Simple Guesses",
        "models": run_baselines(training_records, test_sets, args.max_iterations),
        "source": "Measured by running this program on the supplied story dataset.",
        "notes": [
            "The most-common-answer system ignores the story text.",
            "The word-count system ignores word order.",
            "The vocabulary and both models are fit on training examples only.",
        ],
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    args.results_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    for name, model in result["models"].items():
        print(name)
        for test, measured in model["test_results"].items():
            print(f"  {test}: {measured['accuracy']:.2%} ({measured['correct']}/{measured['examples']})")
    print(f"Saved results to {args.results_file}")


if __name__ == "__main__":
    main()

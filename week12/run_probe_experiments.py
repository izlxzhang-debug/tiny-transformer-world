#!/usr/bin/env python3
"""Run PCA, probing controls, untrained-model controls, and word baselines.

Expected inputs are the three .pt files made by extract_hidden_states.py.
The transformer remains frozen throughout; only the small probes are trained.
"""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from week12.extract_hidden_states import (  # noqa: E402
    LABELS,
    TinyTransformerWithHiddenStates,
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


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()

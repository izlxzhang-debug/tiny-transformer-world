"""Check the published experiment layout and standalone execution."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    "1. Predict from Answer and Word Frequency": "predict_from_frequency.py",
    "2. Memorize 100 Stories": "memorize_100_stories.py",
    "3. Train with Three Random Seeds": "train_with_three_seeds.py",
    "4. Test on Unfamiliar Stories": "test_unfamiliar_stories.py",
    "5. Analyze Errors and Compare Model Changes": "analyze_errors_and_model_changes.py",
    "6. Read World States from the Model": "read_world_states.py",
    "7. Change Internal Values and Test Answers": "test_internal_changes.py",
}


def run(program, arguments, cwd):
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(cwd / "matplotlib-cache"),
    }
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(program), *map(str, arguments)],
        cwd=cwd, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("folder,filename", TRIALS.items())
def test_each_experiment_is_one_program_and_one_results_file(tmp_path, folder, filename):
    directory = ROOT / "Training and Testing Transformer" / folder
    assert {p.name for p in directory.iterdir()} == {filename, "results.json"}
    result = json.loads((directory / "results.json").read_text())
    assert result["experiment"] == folder
    source = directory / filename
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("week", "tiny_transformer_world"))
    copied = tmp_path / filename
    shutil.copyfile(source, copied)
    assert "usage:" in run(copied, ["--help"], tmp_path).stdout.lower()


def test_small_pipeline_runs_from_copied_programs(tmp_path):
    """Exercise all stages without overwriting published or original results."""
    programs = {}
    for folder, filename in TRIALS.items():
        destination = tmp_path / filename
        shutil.copyfile(ROOT / "Training and Testing Transformer" / folder / filename, destination)
        programs[folder] = destination
    data = ROOT / "Dataset/5. Generate Full Dataset/dataset.json"
    data_dir = data.parent
    small_architecture = ["--d-model", "8", "--heads", "2", "--ff-dim", "16"]

    learning_report = tmp_path / "learning.json"
    run(programs["2. Memorize 100 Stories"], [
        "--data", data, "--examples", "20", "--epochs", "1", "--layers", "1",
        *small_architecture, "--target-accuracy", "0", "--device", "cpu",
        "--checkpoint", tmp_path / "overfit.pt", "--results-file", learning_report,
    ], tmp_path)
    assert json.loads(learning_report.read_text())["configuration"]["examples"] == 20

    training_dir = tmp_path / "training"
    run(programs["3. Train with Three Random Seeds"], [
        "--train-data", data, "--validation-data", data,
        "--max-train", "20", "--max-validation", "2000",
        "--epochs", "1", "--layers", "1", *small_architecture,
        "--device", "cpu", "--output-dir", training_dir,
        "--results-file", tmp_path / "training.json",
    ], tmp_path)
    checkpoint = training_dir / "seed_11/checkpoint_best.pt"
    assert checkpoint.is_file()

    run(programs["4. Test on Unfamiliar Stories"], [
        "--data-dir", data_dir, "--checkpoint-root", training_dir,
        "--max-per-condition", "20", "--max-baseline-train", "100",
        "--device", "cpu", "--output-dir", tmp_path / "evaluation",
        "--results-file", tmp_path / "evaluation.json",
    ], tmp_path)

    run(programs["5. Analyze Errors and Compare Model Changes"], [
        "--data-dir", data_dir, "--train-limit", "40", "--validation-limit", "20",
        "--test-limit", "20", "--fewer-examples", "20", "--epochs", "1",
        *small_architecture, "--device", "cpu", "--output-dir", tmp_path / "errors",
        "--results-file", tmp_path / "errors.json",
    ], tmp_path)

    hidden_dir = tmp_path / "hidden"
    inside_report = tmp_path / "inside.json"
    run(programs["6. Read World States from the Model"], [
        "--checkpoint", checkpoint, "--data", data, "--max-examples", "200",
        "--max-train", "200", "--device", "cpu", "--hidden-dir", hidden_dir,
        "--output-dir", tmp_path / "probes", "--results-file", inside_report,
    ], tmp_path)
    inside = json.loads(inside_report.read_text())
    assert set(inside["hidden_state_metadata"]) == {"train", "validation", "test_standard"}
    assert all(m["examples"] == 200 for m in inside["hidden_state_metadata"].values())
    assert (tmp_path / "probes/probe_summary.csv").is_file()

    run(programs["7. Change Internal Values and Test Answers"], [
        "--checkpoint", checkpoint, "--controls", "1", "--top-k", "1",
        "--device", "cpu", "--output-dir", tmp_path / "interventions",
        "--results-file", tmp_path / "interventions.json",
    ], tmp_path)
    assert (tmp_path / "interventions.json").is_file()

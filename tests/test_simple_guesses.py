"""Check that both baselines remain independent of transformer code/test fitting."""
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "Training and Testing Transformer/Simple Guesses/simple_guesses.py"


def test_prediction_summary_counts_errors_by_length():
    module = runpy.run_path(str(PROGRAM))
    records = [
        {"answer": "Lammy", "metrics": {"story_length": 1}},
        {"answer": "Jade", "metrics": {"story_length": 2}},
    ]
    measured = module["evaluate_predictions"](records, ["Lammy", "Lammy"], ["Lammy", "Jade"])
    assert measured["accuracy"] == 0.5
    assert measured["confusion_matrix"] == [[1, 0], [1, 0]]
    assert measured["accuracy_by_story_length"]["1"]["accuracy"] == 1
    assert measured["accuracy_by_story_length"]["2"]["accuracy"] == 0


def test_copied_program_runs_both_baselines_and_fits_only_training_words(tmp_path):
    folder = tmp_path / "program"
    folder.mkdir()
    program = folder / PROGRAM.name
    shutil.copyfile(PROGRAM, program)
    labels = ["Lammy", "Anneena", "Jade", "Penguin", "Nobody"]
    def records(split, extra=""):
        return [
            {"story_text": f"{label} carries key {extra}", "answer": label,
             "metrics": {"story_length": 1}, "split": split}
            for _ in range(4) for label in labels
        ]
    bundle = {"train": records("train")}
    for split in ("test_standard", "test_long", "test_paraphrase", "test_withheld"):
        bundle[split] = records(split, "unseenmarker")
    data = tmp_path / "dataset.json"
    data.write_text(json.dumps({"splits": bundle}))
    process = subprocess.run(
        [sys.executable, "-I", "-B", str(program), "--data", str(data)],
        cwd=tmp_path, env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert {p.name for p in folder.iterdir()} == {PROGRAM.name, "results.json"}
    result = json.loads((folder / "results.json").read_text())
    assert result["experiment"] == "Simple Guesses"
    majority = result["models"]["most_common_answer"]
    words = result["models"]["word_counts"]
    assert majority["majority_label"] == "Lammy"
    assert words["vocabulary_size"] == 7  # Five names plus 'carries' and 'key'.
    for split in ("standard", "long", "paraphrase", "withheld"):
        assert majority["test_results"][split]["accuracy"] == 0.2
        assert words["test_results"][split]["accuracy"] == 1.0

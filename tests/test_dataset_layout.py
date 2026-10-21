"""Check standalone weekly programs and separation of packaged datasets."""

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    2: "1. Generate Example Stories",
    3: "2. Test World Rules",
    4: "3. Generate 300 Stories",
    5: "4. Generate Training and Test Sets",
    6: "5. Generate Full Dataset",
}
PROGRAMS = [
    (2, "generate_example_stories.py", []),
    (3, "test_world_rules.py", []),
    (4, "generate_300_stories.py", []),
    (5, "generate_training_and_test_sets.py", [
        "--train-size", "20", "--validation-size", "20", "--test-size", "20",
    ]),
    (6, "generate_full_dataset.py", [
        "--train-size", "20", "--split-size", "20", "--manual-size", "20",
        "--progress-every", "0",
    ]),
]


@pytest.mark.parametrize("week,filename,arguments", PROGRAMS)
def test_weekly_program_runs_with_only_its_own_file(
    tmp_path, week, filename, arguments,
):
    """Copy no dependencies and run outside the repository with isolated Python."""
    source = ROOT / "Dataset" / FOLDERS[week] / filename
    program = tmp_path / filename
    shutil.copyfile(source, program)
    subprocess.run(
        [sys.executable, "-I", "-B", str(program), *arguments],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert {p.name for p in tmp_path.iterdir()} == {filename, "dataset.json"}
    data = json.loads((tmp_path / "dataset.json").read_text())
    if week == 2:
        assert len(data) == 11
        assert data[0]["answer"] == "Lammy"
    elif week == 3:
        assert len(data["state_trace"]) == 5
        assert [q["answer"] for q in data["questions"]] == ["Anneena", "Lammy"]
    elif week == 4:
        assert len(data) == 300
    else:
        expected = {"train", "validation", "test"} if week == 5 else {
            "train", "validation", "test_standard", "test_long",
            "test_paraphrase", "test_withheld",
        }
        assert set(data["splits"]) == expected
        ids = set()
        for split, records in data["splits"].items():
            assert len(records) == 20
            assert all(record["split"] == split for record in records)
            current_ids = {record["story_id"] for record in records}
            assert len(current_ids) == len(records)
            assert ids.isdisjoint(current_ids)
            ids.update(current_ids)


@pytest.mark.parametrize("filename", [
    "Training and Testing Transformer/2. Memorize 100 Stories/memorize_100_stories.py",
    "Training and Testing Transformer/3. Train with Three Random Seeds/train_with_three_seeds.py",
    "Training and Testing Transformer/4. Test on Unfamiliar Stories/test_unfamiliar_stories.py",
    "Training and Testing Transformer/5. Analyze Errors and Compare Model Changes/analyze_errors_and_model_changes.py",
    "Training and Testing Transformer/6. Read World States from the Model/read_world_states.py",
    "Training and Testing Transformer/1. Predict from Answer and Word Frequency/predict_from_frequency.py",
])
def test_consumers_select_only_the_requested_split(tmp_path, filename):
    bundle = tmp_path / "dataset.json"
    splits = {
        "train": [{"story_id": "train_only", "split": "train"}],
        "validation": [{"story_id": "validation_only", "split": "validation"}],
        "test_standard": [{"story_id": "test_only", "split": "test_standard"}],
    }
    bundle.write_text(json.dumps({"splits": splits}))
    reader = runpy.run_path(str(ROOT / filename))["iter_dataset_records"]
    for split, expected in splits.items():
        assert list(reader(bundle, split=split)) == expected
    with pytest.raises(ValueError, match="Unknown dataset split"):
        list(reader(bundle, split="missing"))

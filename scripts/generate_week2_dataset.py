"""Generate the audited Week 2 little-world dataset."""

from __future__ import annotations

import json
from pathlib import Path

from tiny_transformer_world.week2_dataset import build_week2_dataset


def main() -> None:
    dataset = build_week2_dataset()
    output_path = Path("data/week2_little_world_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(dataset)} stories to {output_path}")
    for record in dataset:
        print(
            f"{record['story_id']}: {record['answer']} "
            f"(length={record['metrics']['story_length']}, "
            f"depth={record['metrics']['reasoning_depth']}, "
            f"evidence={record['evidence_event_ids']})"
        )


if __name__ == "__main__":
    main()

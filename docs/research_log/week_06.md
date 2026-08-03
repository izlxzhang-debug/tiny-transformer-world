# Week 6 Research Log: Full and Specialized Datasets

**Status:** Generator implemented and verified

**Last updated:** 3 August 2026

## Output

[`scripts/generate_week6_full_dataset.py`](../../scripts/generate_week6_full_dataset.py)
is one complete, independently runnable, standard-library-only program. Its
default output contains 20,000 training examples and five 2,000-example
validation or test splits: validation, standard, long-story, paraphrase, and
withheld-combination.

The program validates every story by replay, enforces exact answer and queried-
object balance, prevents cross-split duplicates, and confirms that four defined
pickup combinations are absent from training and covered by the specialized
withheld test. It also exports 100 balanced examples for genuine manual review;
the program deliberately does not claim that the human review is complete.

## Run

```bash
python3 scripts/generate_week6_full_dataset.py
```

The generated JSONL files, statistics, validation report, and human-review CSV
are written under `data/week6_full_dataset/` by default.

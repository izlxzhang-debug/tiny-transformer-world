# Week 6 Research Log: Full and Specialized Datasets

**Status:** Generator implemented and verified

**Last updated:** 3 August 2026

## Output

[`Dataset/Training and Harder Tests/generate_full_dataset.py`](../../Dataset/Training%20and%20Harder%20Tests/generate_full_dataset.py)
is one complete, independently runnable, standard-library-only program. Its
default output contains 20,000 training examples and five 2,000-example
validation or test splits: validation, standard, long-story, paraphrase, and
withheld-combination.

The program validates every story by replay, enforces exact answer and queried-
object balance, prevents cross-split duplicates, and confirms that four defined
pickup combinations are absent from training and covered by the specialized
withheld test. It also selects 100 balanced example references for manual review;
the program deliberately does not claim that the human review is complete.

## Run

```bash
python3 'Dataset/Training and Harder Tests/generate_full_dataset.py'
```

The six datasets are stored under `splits` in `Dataset/Training and Harder Tests/dataset.json`.
That same file holds the validation report and the 100 manual-review references
with blank status and notes fields. Training and evaluation readers select
the requested split without mixing training, validation, and test examples.

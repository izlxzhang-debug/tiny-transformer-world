# Week 5 Research Log: Complete Dataset Generator

**Status:** Generator implemented and verified

**Last updated:** 3 August 2026

## Output

[`scripts/generate_week5_dataset.py`](../../scripts/generate_week5_dataset.py)
is one complete, independently runnable, standard-library-only program. By
default it creates 10,000 training, 1,000 validation, and 1,000 standard test
examples.

Every split is jointly balanced across four queried objects and five answer
classes. The program rejects duplicate structured stories across splits,
replays every accepted story, checks its state and answer, and reports 100%
rule-based accuracy with zero invalid actions before saving the data.

## Run

```bash
python3 scripts/generate_week5_dataset.py
```

Reduced sizes can be supplied through the documented command-line options for
quick verification. All split sizes must be positive multiples of 20.

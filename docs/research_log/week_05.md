# Week 5 Research Log: Complete Dataset Generator

**Status:** Generator implemented and verified

**Last updated:** 3 August 2026

## Output

[`Dataset/4. Generate Training and Test Sets/generate_training_and_test_sets.py`](../../Dataset/4.%20Generate%20Training%20and%20Test%20Sets/generate_training_and_test_sets.py)
is one complete, independently runnable, standard-library-only program. By
default it creates 10,000 training, 1,000 validation, and 1,000 standard test
examples.

The three splits and their statistics are packaged in the adjacent
`dataset.json`, under `splits` and `statistics`. Each weekly folder contains
only its standalone program and this data file.

Every split is jointly balanced across four queried objects and five answer
classes. The program rejects duplicate structured stories across splits,
replays every accepted story, checks its state and answer, and reports 100%
rule-based accuracy with zero invalid actions before saving the data.

## Run

```bash
python3 'Dataset/4. Generate Training and Test Sets/generate_training_and_test_sets.py'
```

Reduced sizes can be supplied through the documented command-line options for
quick verification. All split sizes must be positive multiples of 20.

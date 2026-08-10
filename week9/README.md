# Week 9: Three-Seed Tiny-Transformer Training

This folder contains the complete standalone program for training the
word-level tiny transformer with three random seeds and comparing the runs.

## Included file

- `train_three_seed_transformer.py`: model, tokenizer, training loop,
  validation loop, checkpoint system, metrics export, and diagnostics in one
  runnable Python file.

The program uses only `story_text` as input and predicts one of the five
carrier answers: Lammy, Anneena, Jade, Penguin, or Nobody. It learns the
vocabulary from the training split only and maps unseen validation words to
`<unk>`.

## Full run

Run from the repository root:

```bash
source .venv/bin/activate
python3 week9/train_three_seed_transformer.py
```

The defaults use:

- 20,000 training stories;
- 2,000 validation stories;
- random seeds 11, 22, and 33;
- 20 epochs per seed;
- two transformer layers;
- model dimension 64;
- two attention heads;
- feed-forward dimension 128; and
- automatic CUDA, Apple MPS, or CPU selection.

## Fast verification

```bash
python3 week9/train_three_seed_transformer.py \
  --device cpu \
  --epochs 2 \
  --max-train 200 \
  --max-validation 100 \
  --output-dir results/tiny_transformer_training_smoke
```

## Saved outputs

The default output directory is `results/tiny_transformer_training/`.
Generated results and checkpoints are intentionally excluded from Git because
the checkpoints are reproducible binary artifacts.

The program creates:

```text
results/tiny_transformer_training/
├── class_balance.json
├── diagnostics.json
├── run_configuration.json
├── seed_summary.csv
├── summary.json
├── vocabulary.json
├── seed_11/
│   ├── checkpoint_best.pt
│   ├── checkpoint_final.pt
│   ├── diagnostics.json
│   ├── history.csv
│   ├── history.json
│   └── settings.json
├── seed_22/
└── seed_33/
```

Each epoch records training loss, validation loss, training accuracy,
validation accuracy, and duration. Training metrics are recomputed after each
epoch with fixed weights and dropout disabled, making them comparable with the
validation metrics.

## Completed full-run results

| Seed | Final train accuracy | Final validation accuracy | Best validation accuracy | Duration |
|---:|---:|---:|---:|---:|
| 11 | 70.76% | 69.15% | 69.70% | 62.25 min |
| 22 | 71.43% | 69.45% | 70.45% | 59.37 min |
| 33 | 71.21% | 68.80% | 70.35% | 59.06 min |

Across the three seeds:

- mean final validation accuracy: 69.13%;
- population standard deviation: 0.27 percentage points;
- final validation accuracy range: 0.65 percentage points;
- total training duration: approximately 3.01 hours;
- class imbalance detected: no;
- overfitting detected by the defined heuristics: no;
- unstable training detected: no; and
- large seed-to-seed differences detected: no.

Chance accuracy for the five balanced answer classes is 20%. The full-run
validation accuracy is therefore substantially above chance, but behavioral
accuracy alone does not establish that the model learned a reusable internal
world model.

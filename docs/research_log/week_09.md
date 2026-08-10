# Week 9 Research Log: Three-Seed Full Training

**Status:** Complete

**Last updated:** 10 August 2026

## Work completed

- Trained the word-level tiny transformer on all 20,000 training stories.
- Evaluated all 2,000 validation stories after every epoch.
- Repeated the experiment with random seeds 11, 22, and 33 while keeping the
  data split and model settings fixed.
- Saved model settings, random seed, per-epoch training and validation loss,
  per-epoch accuracy, epoch duration, total duration, best checkpoint, and
  final checkpoint for every seed.
- Added automatic checks for overfitting, unstable training, class imbalance,
  and large differences between random seeds.
- Kept the complete runnable implementation in
  [`week9/train_three_seed_transformer.py`](../../week9/train_three_seed_transformer.py).

## Model settings

- Two transformer layers
- Model dimension 64
- Two attention heads, each with dimension 32
- Feed-forward dimension 128
- Dropout 0.1
- Batch size 64
- AdamW learning rate 0.001
- Weight decay 0.0001
- 20 epochs per seed
- Five output classes: Lammy, Anneena, Jade, Penguin, and Nobody

## Results

| Seed | Final training accuracy | Final validation accuracy | Best validation accuracy |
|---:|---:|---:|---:|
| 11 | 70.76% | 69.15% | 69.70% |
| 22 | 71.43% | 69.45% | 70.45% |
| 33 | 71.21% | 68.80% | 70.35% |

Mean final validation accuracy was 69.13%. The population standard deviation
was 0.27 percentage points, and the difference between the highest and lowest
final validation accuracies was 0.65 percentage points.

## Diagnostic interpretation

- **Overfitting:** Not detected. Final training-validation accuracy gaps were
  far below the 10-point threshold, and validation loss did not show the
  defined deterioration pattern while training loss continued falling.
- **Unstable training:** Not detected. Losses remained finite and did not show
  the defined large epoch-to-epoch jumps.
- **Class imbalance:** Not detected. Training contained 4,000 examples per
  answer class, and validation contained 400 per class.
- **Random-seed sensitivity:** No large difference was detected. The final
  validation-accuracy range was 0.65 points, below the five-point threshold.

The result establishes repeatable above-chance behavioral learning across
three initializations. It does not by itself show that the transformer learned
a causally used or reusable internal representation of the fictional world.

## Reproduction

From the repository root:

```bash
source .venv/bin/activate
python3 week9/train_three_seed_transformer.py
```

Generated metrics and checkpoints are written under
`results/tiny_transformer_training/` and remain outside version control.

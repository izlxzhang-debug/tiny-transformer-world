# From Prediction to Understanding

This workspace supports the passion project:

> Does a tiny transformer learn a reusable internal model of a controlled
> fictional world, or does it rely on surface-level language patterns?

## Current status

| Week | Focus | Status | Main output |
|---:|---|---|---|
| 1 | Research question and scope | Complete | [Project proposal](docs/proposal.md) |
| 2 | Mathematical foundations | Not started | Mathematics notebook |
| 3 | Attention and literature | Not started | Reading summaries |
| 4 | Formal world design | Not started | Formal specification |

Week 1 also produced a [working glossary](docs/glossary.md), a
[preliminary world specification](docs/world_specification.md), and a
[research log](docs/research_log/week_01.md). The canonical
[research framework](docs/research_framework.md) records the primary question,
three secondary questions, testable hypothesis, five predictions, and
operational definition used by the experiments.

## What is installed

- PyTorch for the tiny transformer and GRU/LSTM baseline
- NumPy and pandas for data generation and analysis
- scikit-learn for bag-of-words baselines, PCA, and linear probes
- Matplotlib and seaborn for figures
- JupyterLab for the mathematics and results notebooks
- pytest for simulator and dataset validation

## Start working

Open a terminal in this folder and run:

```bash
source .venv/bin/activate
python scripts/check_environment.py
jupyter lab
```

Leave the environment with:

```bash
deactivate
```

## Project layout

```text
data/          generated datasets and special test sets
docs/          proposal, definitions, specifications, and weekly research logs
src/           simulator, models, training, evaluation, and interpretability code
tests/         simulator and dataset correctness tests
notebooks/     mathematics, behavioral results, and representation analysis
figures/       final charts and causal heatmaps
checkpoints/   trained model checkpoints
results/       experiment metrics and tables
paper/         proposal, research paper, and philosophy essay
scripts/       environment and utility commands
```

## Recommended build order

1. Write the one-page proposal and operational definition of “world model.”
2. Implement and test the formal fictional-world simulator.
3. Generate and validate balanced datasets.
4. Establish majority-class and bag-of-words baselines.
5. Train the tiny transformer with three random seeds.
6. Run generalization tests, hidden-state probes, and controls.
7. Attempt activation patching as the stretch experiment.
8. Interpret the results in the research paper and philosophy essay.

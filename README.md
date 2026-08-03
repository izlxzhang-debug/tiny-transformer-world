# From Prediction to Understanding

This workspace supports the passion project:

> Does a tiny transformer learn a reusable internal model of a controlled
> fictional world, or does it rely on surface-level language patterns?

## Current status

| Week | Focus | Status | Main output |
|---:|---|---|---|
| 1 | Research question and scope | Complete | [Project proposal](docs/proposal.md) |
| 2 | Formal world construction | Complete | [World simulator](src/tiny_transformer_world/world.py) and [dataset](data/week2_little_world_dataset.json) |
| 3 | Standalone simulator verification | Complete | [Copy-and-run simulator](week3_simulator.py) and [research log](docs/research_log/week_03.md) |
| 4 | Balanced dataset generation | Complete | [Complete generator](generate_300_stories.py) and [300 validated stories](data/world_stories_300.json) |
| 5 | Train/validation/test generation | Generator ready | [Standalone Week 5 generator](scripts/generate_week5_dataset.py) and [research log](docs/research_log/week_05.md) |
| 6 | Full and specialized test generation | Generator ready | [Standalone Week 6 generator](scripts/generate_week6_full_dataset.py) and [research log](docs/research_log/week_06.md) |

Week 1 also produced a [working glossary](docs/glossary.md), a
[preliminary world specification](docs/world_specification.md), and a
[research log](docs/research_log/week_01.md). The canonical
[research framework](docs/research_framework.md) records the primary question,
three secondary questions, testable hypothesis, five predictions, and
operational definition used by the experiments.

Week 2 implemented the deterministic simulator, eleven audited coverage
stories, complete JSON records with state traces and evidence IDs, and
automated tests. See the [Week 2 research log](docs/research_log/week_02.md).

Week 3 produced one self-contained Python file containing the simulator,
edge-case checks, and a complete example. It uses only the Python standard
library and does not require the package or any other project file. Copy
[week3_simulator.py](week3_simulator.py) into a Python file and run it with:

```bash
python3 week3_simulator.py
```

The program should print `All tests passed.` before showing the five world
states and final answers.

Week 4 keeps the simulator, generator, validation, statistics, and runnable
entry point together in one complete Python file. It produces 300 stories
balanced across all five answers and all four queried objects. See the
[Week 4 research log](docs/research_log/week_04.md).

The Week 5 and Week 6 generators are also complete, independently runnable
Python files. Week 5 produces balanced train, validation, and standard test
splits. Week 6 produces the full 30,000-example dataset, including long-story,
paraphrase, and withheld-combination test sets plus a 100-example human-review
sheet. Generated Week 5 and Week 6 data are reproducible outputs and are not
committed to the repository.

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
python3 week3_simulator.py
python3 generate_300_stories.py
python3 scripts/generate_week5_dataset.py
python3 scripts/generate_week6_full_dataset.py
python scripts/check_environment.py
python scripts/generate_week2_dataset.py
pytest
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

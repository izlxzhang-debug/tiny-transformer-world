# From Prediction to Understanding

This workspace supports the passion project:

> Does a tiny transformer learn a reusable internal model of a controlled
> fictional world, or does it rely on surface-level language patterns?

## Current status

| Week | Focus | Status | Main output |
|---:|---|---|---|
| 1 | Research question and scope | Complete | [Project proposal](docs/proposal.md) |
| 2 | Formal world construction | Complete | [Standalone program](Dataset/Example%20Stories/generate_examples.py) and [dataset](Dataset/Example%20Stories/dataset.json) |
| 3 | Standalone simulator verification | Complete | [Copy-and-run simulator](Dataset/Testing%20the%20Rules/check_rules.py) and [research log](docs/research_log/week_03.md) |
| 4 | Balanced dataset generation | Complete | [Complete generator](Dataset/300%20Stories/generate_samples.py) and [300 validated stories](Dataset/300%20Stories/dataset.json) |
| 5 | Train/validation/test generation | Dataset included | [Standalone generator](Dataset/Training%20and%20Test%20Stories/generate_splits.py) and [dataset](Dataset/Training%20and%20Test%20Stories/dataset.json) |
| 6 | Full and specialized test generation | Dataset included | [Standalone generator](Dataset/Training%20and%20Harder%20Tests/generate_full_dataset.py) and [dataset](Dataset/Training%20and%20Harder%20Tests/dataset.json) |
| 8 | Tiny-transformer construction | Overfitting test passed | [Tiny transformer](Training%20and%20Testing%20Transformer/Learning%20100%20Stories/learn_100_stories.py) and [research log](docs/research_log/week_08.md) |
| 9 | Three-seed full training | Complete | [Week 9 package](docs/training_instructions.md) and [research log](docs/research_log/week_09.md) |
| 10 | Generalization tests | Recorded results included | [Testing New Stories](Training%20and%20Testing%20Transformer/Testing%20New%20Stories/) |
| 11 | Error analysis and model changes | Recorded results included | [Finding Mistakes](Training%20and%20Testing%20Transformer/Finding%20Mistakes/) |
| 12 | Hidden vectors and probes | Recorded results included | [Looking Inside the Model](Training%20and%20Testing%20Transformer/Looking%20Inside%20the%20Model/) |
| 13 | Activation interventions | Recorded results included | [Testing What the Model Uses](Training%20and%20Testing%20Transformer/Testing%20What%20the%20Model%20Uses/) |

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
[Dataset/Testing the Rules/check_rules.py](Dataset/Testing%20the%20Rules/check_rules.py) into a Python file and run it with:

```bash
python3 'Dataset/Testing the Rules/check_rules.py'
```

The program should print `All tests passed.` before showing the five world
states and final answers. It also saves that worked example, including its
state trace and two questions, in the adjacent `dataset.json`.

Week 4 keeps the simulator, generator, validation, statistics, and runnable
entry point together in one complete Python file. It produces 300 stories
balanced across all five answers and all four queried objects. See the
[Week 4 research log](docs/research_log/week_04.md).

The Week 5 and Week 6 generators are also complete, independently runnable
Python files. Week 5 produces balanced train, validation, and standard test
splits. Week 6 produces the full 30,000-example dataset, including long-story,
paraphrase, and withheld-combination test sets plus 100 example references for
human review. The original generated datasets are included in the repository.

The five folders under `Dataset/` are named for their purpose:

| Folder | Purpose | Original stage |
|---|---|---|
| `Example Stories` | Eleven worked stories with correct world states | Week 2 |
| `Testing the Rules` | Simulator rule checks and a worked example | Week 3 |
| `300 Stories` | Generate 300 balanced sample stories | Week 4 |
| `Training and Test Stories` | Generate training, validation, and standard test sets | Week 5 |
| `Training and Harder Tests` | Generate the full dataset and specialized tests | Week 6 |

Each folder contains exactly one standalone Python program and one
`dataset.json`. Run any program
with Python 3.11 or newer; no other project file or third-party package is
needed. By default its data file is saved alongside that program.

Week 5 and Week 6 store distinct datasets under the JSON object's `splits` key.
Week 5 has `train`, `validation`, and `test`; Week 6 has `train`, `validation`,
`test_standard`, `test_long`, `test_paraphrase`, and `test_withheld`. Statistics
and the validation report are stored in the same file. Combining the files
does not combine the training and test examples into one split.

Training and evaluation programs automatically read the requested split from
`Dataset/Training and Harder Tests/dataset.json`. Their readers also support the earlier separate
JSONL files when those files are explicitly supplied. Existing simulator
modules under `src/` remain available for package imports and automated tests.

Week 8 implements a self-contained word-level PyTorch transformer with learned
token and positional embeddings, explicit multi-head Q/K/V attention, padding
masks, one or two transformer layers, carrier classification, and resumable
checkpointing. Its balanced 100-example smoke test reached 99% memorization
accuracy. This confirms that the training pipeline can fit a tiny dataset; it
is not evidence of generalization or a learned reusable world model. See the
[Week 8 research log](docs/research_log/week_08.md).

Week 9 trains the tiny transformer on the complete 20,000-story training split
with seeds 11, 22, and 33, evaluates all 2,000 validation stories after every
epoch, saves best and final checkpoints, and checks overfitting, instability,
class balance, and seed sensitivity. Mean final validation accuracy was 69.13%
with a 0.65-point range across seeds. See the [Week 9 package](docs/training_instructions.md).

## Training and Testing Transformer

Each of these folders contains one complete Python program and one
`results.json` containing the results already recorded for that experiment:

| Folder | What the program does | Original stage |
|---|---|---|
| `Learning 100 Stories` | Check whether the model can memorize a small set | Week 8 |
| `Training the Model` | Train with three random seeds and compare the runs | Week 9 |
| `Testing New Stories` | Test unfamiliar stories and compare with baselines | Week 10 |
| `Finding Mistakes` | Examine wrong answers and test changes to the model | Week 11 |
| `Looking Inside the Model` | Extract internal vectors and test what can be read from them | Week 12 |
| `Testing What the Model Uses` | Change internal activations and measure the effect on answers | Week 13 |

The programs include their model and analysis code without importing Python
files from other project weeks. They require the installed libraries, the
relevant story data, and, for later experiments, a trained checkpoint.
Checkpoints and large hidden-vector files remain under `checkpoints/` and
`results/` locally. They are referenced in the combined reports rather than
uploaded inside these folders. On a fresh clone, run `Training the Model` to
create the checkpoints needed for the later experiments.

Existing JSON reports and CSV tables are preserved inside each `results.json`.
The small memorization report comes from its saved checkpoint, which records
final accuracy but not a per-epoch loss history. New runs refresh the adjacent
report by default; use `--results-file` to save a new report elsewhere. Raw
checkpoints, plots, and tables keep their existing output directories.

`Looking Inside the Model` combines the old extraction and probe programs.
Its default command runs both stages; `extract --help` and `probe --help`
describe the individual stages. Probe scores show what information can be
recovered and do not by themselves establish that the model uses it.

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
python3 'Dataset/Testing the Rules/check_rules.py'
python3 'Dataset/300 Stories/generate_samples.py'
python3 'Dataset/Training and Test Stories/generate_splits.py'
python3 'Dataset/Training and Harder Tests/generate_full_dataset.py'
python3 'Training and Testing Transformer/Learning 100 Stories/learn_100_stories.py'
python3 'Training and Testing Transformer/Training the Model/train_model.py'
python scripts/check_environment.py
python 'Dataset/Example Stories/generate_examples.py'
pytest
jupyter lab
```

Leave the environment with:

```bash
deactivate
```

## Project layout

```text
Dataset/       Weeks 2-6: one standalone .py and one dataset.json per week
Training and Testing Transformer/  Experiments: one complete .py and one results.json each
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

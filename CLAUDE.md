# CLAUDE.md

Project: multilabel image classification, 5 classes, scored by Macro F1. Full task and submission format: `docs/overview.md`. Competition rules: `docs/rules.md` (no external data, no relabeling, pretrained weights only from timm or ImageNet).

- Never execute notebooks at full scale.
- Notebooks should not contain much text.

## Directories

- `core/`: only Python package, all project code, one module per section when section needs code. Installed by `poetry install`, imported as `core.<module>` from any notebook, no `sys.path` edits.
- `sections/<section>/`: sections in workflow order: `eda`, `validation`, `baseline`, `augmentation`, `model`, `ensemble`. Section name = folder name.
- `data/kaggle/`: Kaggle zip as is, read-only. `data/processed/`: generated files. `data/submissions/`: CSVs for upload. All gitignored.
- `docs/`: task, rules, backlog. `docs/course/`: course materials, gitignored.

Each section holds, folders created only when first file lands:

- Report: `<section folder>/<section>_report.md`
- Experiment cards: `<section folder>/experiments/`
- Experiment notebooks: `<section folder>/notebooks/experiments/`
- Submission notebooks: `<section folder>/notebooks/submissions/`
- Figures: `<section folder>/figures/`

## Experiments

- Tracking prefix: `cnn` (MLflow experiments `cnn-<section>-<slug>`, database `mlflow.db` at the project root)
- Jupyter: `.venv/bin/jupyter`

# Project structure

## Directories

- `src/`: all project code, three packages. Installed by `poetry install`, imported as `<package>.<module>` from any notebook, no `sys.path` edits.
  - `dataset/`: datasets, folds (`dataset`), transforms (`augment`).
  - `training/`: paths and constants (`config`), `cli` (LightningCLI, one run per fold), `runs` (run folder format: paths, writer, loader), `cv` (`CVResult`), `train` (callbacks), `ensemble` (mean of CV runs), `submission` (CSV of one model or ensemble), `model`, `tracking`.
  - `analysis/`: `metrics` (AP, PR curve), `thresholds` (best, plateau, cross-fold F1), `gradcam`, `plots`.
- `configs/`: training configs, `expNN_<slug>/<model>.yaml`, one per model.
- `notebooks/`: `experiments/` (one per card), `submissions/`.
- `data/kaggle/`: Kaggle zip as is, read-only. `data/processed/`: generated files. `data/submissions/`: CSVs for upload. All gitignored.
- `docs/`: `task/` (task, rules), `figures/` (all figures). `docs/course/`: course materials, gitignored.
  - `neuroslope/`: all AI-written docs. `methods/` (topic docs), `reports/` (`<topic>_report.md`: eda, validation, model, ensemble), `experiments/` (cards), `ai_usage/` (AI chat transcripts), `code notes/`, `backlog.md`, `theory_answers.md`.

## Branches

- `playground`: scratch, cut from `main`, notebooks and configs straight in `notebooks/`, `configs/`. Never merged into `main`.
- `playground-archive`: orphan, old scratch files, read-only.

## Experiment files

One numbering for all experiments, folders created only when first file lands:

- Card: `docs/neuroslope/experiments/expNN_<slug>.md`
- Notebook: `notebooks/experiments/expNN_<slug>.ipynb`
- Configs: `configs/expNN_<slug>/<model>.yaml`, LightningCLI: top `name`, `experiment` plus `model` (`LitCNNMultiLabel`), `data` (`KaggleDataModule`), `trainer` sections; blocks `class_path` + `init_args`
- Figures: `docs/figures/expNN_<slug>*.png`

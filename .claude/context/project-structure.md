# Project structure

## Directories

- `src/`: all project code, two packages. Installed by `poetry install`, imported as `<package>.<module>` from any notebook, no `sys.path` edits.
  - `core/`: code affecting training and decisions (what gets predicted, submitted). `cli` (LightningCLI, one run per fold), datasets, folds (`dataset`), transforms (`augment`), `model` (LightningModule, mAP and checkpoint callbacks), `metrics` (AP, PR curve, best, plateau, cross-fold F1 thresholds), `predict` (submission thresholds, test probs over folds, ensemble mean).
  - `helpers/`: code only easing project code: plumbing, tracking, display, figures. Paths and constants (`config`), `tracking` (run folder format: paths, writer, loader; MLflow logger, config saving), `progress` (epoch bars), `cv` (`CVResult`), `ensemble` (mean of CV runs), `submission` (CSV of one model or ensemble, Kaggle upload), `gradcam`, `plots`.
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

# Project structure

## Directories

- `src/core/`: only project package, all project code. Installed by `poetry install`, imported as `core.<package>.<module>` from any notebook, no `sys.path` edits.
  - `data/`: datasets, folds, `augment/`.
  - `training/`: paths and constants (`config`), `cli` (LightningCLI, one run per fold), `runs` (run folder format: paths, writer, loader), `cv` (`CVResult`), `train` (callbacks), `ensemble` (mean, stacking of CV runs), `submission` (CSV of one model or ensemble), `model`, `tracking`.
  - `analysis/`: `metrics` (AP, PR curve, F1 at thresholds), `thresholds` (best, plateau, cross-fold F1), `gradcam`, `plots`.
  - `boxes/`: experimental Grad-CAM pseudo-boxes (exp06–exp24), unused later; main code never imports it.
- `configs/models/`: training configs, `expNN_<slug>/<model>.yaml`, one per model. `configs/models/playground/`: scratch, gitignored.
- `notebooks/`: `experiments/` (one per card), `submissions/`, `playground/` (scratch, gitignored).
- `tests/`: unit tests.
- `data/kaggle/`: Kaggle zip as is, read-only. `data/processed/`: generated files. `data/submissions/`: CSVs for upload. All gitignored.
- `docs/`: `task/` (task, rules), `methods/` (topic docs), `reports/` (`<topic>_report.md`: eda, validation, model, ensemble), `experiments/` (cards), `figures/` (all figures), `backlog.md`. `docs/course/`: course materials, gitignored.

## Experiment files

One numbering for all experiments, folders created only when first file lands:

- Card: `docs/experiments/expNN_<slug>.md`
- Notebook: `notebooks/experiments/expNN_<slug>.ipynb`
- Configs: `configs/models/expNN_<slug>/<model>.yaml`, LightningCLI: top `name`, `experiment` plus `model` (`LitCNNMultiLabel`), `data` (`KaggleDataModule`), `trainer` sections; blocks `class_path` + `init_args`
- Figures: `docs/figures/expNN_<slug>*.png`

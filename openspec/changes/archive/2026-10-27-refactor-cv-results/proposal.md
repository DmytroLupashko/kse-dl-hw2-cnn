# Proposal

## Why

`CVResult` (`src/core/training/cv.py`) holds five jobs: run-folder storage plus legacy loader, scoring, MLflow history, inference, slicing. Run-folder format split: `RunCheckpoint` (`train.py`) writes, `CVResult.load` reads. All 33 finished runs still in legacy format (`cv.json`, pooled `oof.parquet`, checkpoints under `checkpoints/`), so second loader stays forever unless migrated.

## What Changes

- One-off migration: 33 legacy runs rewritten into `fold_<k>/` run folders (checkpoint, `config.yaml`, `oof.parquet`, `run.json`); `checkpoints/`, `cv.json`, pooled `oof.parquet`, unused `fold_ap.parquet` removed after check.
- New `core.training.runs`: run-folder format in one place: `run_dir`, `read_config`, `PROB_COLS`, OOF/`run.json` writer, folder loader.
- `CVResult` keeps data plus scoring, slicing, `fold_model`; `load` delegates to `runs`.
- **BREAKING** (code): `CVResult.load_legacy`, `config.CHECKPOINTS` removed.
- **BREAKING** (code): `CVResult.predict_val` removed; its one notebook gets local helper.
- MLflow curves read by one function in `tracking`; `CVResult.history`, `CVResult.mean_curve` keep names, thin wrappers.
- `fold_ap`, `fold_scores` share one per-fold `Scores` helper; output unchanged.
- `RunCheckpoint` keeps checkpointing, calls `runs` writer.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. Pure refactor plus data migration: `skip_specs: true`.

## Impact

- Code: `src/core/training/{cv,train,cli,tracking,config}.py`, new `src/core/training/runs.py`, `src/core/kaggle/{pull,submission}.py`, `src/core/training/ensemble.py` (import of `PROB_COLS`), one notebook using `predict_val`.
- Data: `data/processed/cv/**`, `checkpoints/` (4.0G moved, not copied). Backup before migration.
- Notebook API otherwise unchanged: `load`, `scores`, `fold_ap`, `fold_scores`, `mean_curve`, `history`, `only`, `fold_model`, `oof`, `config`, `folds`.

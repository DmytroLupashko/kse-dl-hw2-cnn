# Design

## Context

Motivation: see proposal.md.

Legacy run (33 of them): `data/processed/cv/<experiment>/<name>/` holds `cv.json` (`config`, `run_ids` per fold), pooled `oof.parquet` (column `fold`), unused `fold_ap.parquet`; checkpoint in `checkpoints/<experiment>/<name> fold <k>/*.ckpt` next to own `run.json`. Legacy `config` has old `_target_` shape, not LightningCLI shape; consumers read only `experiment`, `name`, `data.{img_size,batch_size,num_workers,n_other}`, all present. `LitCNNMultiLabel.load` already handles old checkpoints.

No finished run in new format yet; `core.kaggle.pull` writes new format.

## Goals / Non-Goals

**Goals:**
- One loader, one place owning run-folder format.
- `CVResult` notebook API stable except `predict_val`.

**Non-Goals:**
- Converting legacy `config` to LightningCLI shape: migrated runs not retrainable from `config.yaml`, only readable.
- `Ensemble`, `Stacking`, `Submission` design.
- Removing `CLASSES` global.

## Decisions

**Module `runs.py` owns format.** Holds `run_dir`, `read_config`, `PROB_COLS`, `save_run(dir, oof, run_id)`, `load_runs(path) -> (config, oof, run_ids, dirs)`. `train.py`, `cli.py`, `kaggle/pull.py`, `ensemble.py`, `submission.py` import from it, not from `cv.py`. Alternative: keep in `cv.py`; rejected, `train` then imports results module for column names.

**`CVResult` stays dataclass.** Fields unchanged. `load` = `cls(*runs.load_runs(path))`. Scoring via private `_fold_scores()` yielding `(fold, Scores)`; `fold_ap`, `fold_scores` built on it.

**History in `tracking`.** `tracking.curves(run_ids, keys)`: long frame (fold, epoch, key, value). `CVResult.history` maps keys to (metric, split) via `HISTORY`; `mean_curve(key)` = epoch mean of `curves`. Alternative: drop `mean_curve`; rejected, 28 notebook calls.

**`predict_val` out.** Only `exp06_augmentations_v3.ipynb` uses it: notebook builds dataset and calls `result.fold_model(fold).predict_probs(...)` inline.

**Migration script one-off, not in `src/core`.** `scripts/migrate_legacy_runs.py`, deleted after run (git history keeps it). Per legacy run, per fold: `fold_<k>/` gets checkpoint moved (`Path.rename`, same disk, 4.0G not duplicated), `run.json` from checkpoint folder, `config.yaml` = legacy `config` dumped as YAML, `oof.parquet` = pooled rows of that fold, index kept. Then `cv.json`, pooled `oof.parquet`, `fold_ap.parquet` deleted, empty `checkpoints/` removed.

## Risks / Trade-offs

- Move breaks results mid-run. Mitigation: backup `data/processed/cv` (50M) as tarball first; script idempotent per fold (skips folder already holding `run.json`); checkpoints moved last per fold.
- Migrated results silently differ. Mitigation: before migration dump `fold_scores()`, `folds`, `oof` per run with old code; after, compare with new code, must be equal.
- Old checkpoint not loadable from new path. Mitigation: `fold_model` smoke load for one run per architecture (AlexNet, ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, RegNetY-016, MobileNetV3).
- Index order of `oof` changes after concat per fold. Accepted: `load_runs` keeps fold order (as new-format loader did); `Ensemble.member_oofs` already sorts; scores order-free.
- One legacy fold without checkpoint (exp42 `cnn-model-oversample-8-only` fold 0, lost before migration): migrated with OOF rows and `run.json` only; `fold_model(0)` fails as before.

## Migration Plan

1. Snapshot with old code (pickled `fold_scores`, `oof`, `folds`, `config` per run).
2. Backup tarball of `data/processed/cv`.
3. Run migration script.
4. New code, compare with snapshot, smoke-load checkpoints.
5. Rollback: restore tarball, move checkpoints back by `run.json` run ids (script prints mapping log).

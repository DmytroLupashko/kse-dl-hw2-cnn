# Tasks

## 1. Snapshot and migration

- [x] 1.1 Snapshot with old code: per run of `data/processed/cv/*/*/`, pickle `fold_scores()`, `oof`, `folds`, `config` to scratchpad; verify 33 legacy runs plus new-format runs in pickle
- [x] 1.2 Backup `data/processed/cv` as tarball in scratchpad; verify `tar -tzf` lists every `cv.json`
- [x] 1.3 Write `scripts/migrate_legacy_runs.py` per design (move checkpoint, copy `run.json`, write `config.yaml`, split `oof.parquet`, idempotent, mapping log); verify dry run prints 33 runs with expected fold counts
- [x] 1.4 Run migration; verify no `cv.json` left, every `fold_<k>/` holds checkpoint, `config.yaml`, `oof.parquet`, `run.json`, `checkpoints/` gone

## 2. `runs.py` owns run-folder format

- [x] 2.1 Create `src/core/training/runs.py` with `run_dir`, `read_config`, `PROB_COLS`, `save_run`, `load_runs`; add `tests/test_runs.py` round trip on tmp folder; verify `pytest tests/test_runs.py` passes
- [x] 2.2 `CVResult.load` via `load_runs`; delete `load_legacy`, `config.CHECKPOINTS`; verify every run's `fold_scores()`, `oof`, `folds`, `config` equal to snapshot
- [x] 2.3 `RunCheckpoint.on_fit_end` calls `save_run`; `cli`, `kaggle/pull`, `ensemble`, `submission` import from `runs`; verify `python -m core.training.cli fit --config <playground config> --data.fold=0 --trainer.max_epochs=1 --trainer.limit_train_batches=2 --trainer.limit_val_batches=2` writes full run folder and `CVResult.load` reads it
- [x] 2.4 Smoke-load `fold_model(0)` for one run per architecture (AlexNet, ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, RegNetY-016, MobileNetV3); verify all load
- [x] 2.5 Update module docstrings (`cv`, `train`, `cli`, `runs`) and `.claude/project-structure.md` `training/` line; verify no reference to `load_legacy`, `CHECKPOINTS`, `cv.run_dir` left (`grep -rn`)

## 3. Scoring dedupe

- [x] 3.1 Private per-fold `Scores` helper; `fold_ap`, `fold_scores` built on it; verify equal to snapshot

## 4. History in `tracking`

- [x] 4.1 `tracking.curves(run_ids, keys)`; `CVResult.history`, `mean_curve` wrap it; update `tracking` docstring; verify both outputs equal old code on one legacy run (values from snapshot run ids)

## 5. `predict_val` out

- [x] 5.1 Remove `CVResult.predict_val`; inline in `notebooks/experiments/exp06_augmentations_v3.ipynb` cell; verify cell source parses and `grep -rn predict_val src notebooks` empty

## 6. Integration

- [x] 6.1 `Single(<config>).thresholds()` and `MeanEnsemble([...], path).oof_probs()` on migrated runs equal values from old code snapshot; `pytest tests` passes
- [x] 6.2 Delete `scripts/migrate_legacy_runs.py` and backup tarball after user confirms; verify `git status` clean of both

## Workflow follow-up

- Archive change after review.

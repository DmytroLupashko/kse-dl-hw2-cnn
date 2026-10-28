# Tasks

## 0. Baseline

- [x] 0.1 Record proof before edits: `.venv/bin/pytest tests` result per test (`test_custom_model.py` excluded: deleted in 2.1; rest must stay as is); script importing every `from core...` line of `notebooks/experiments/*.ipynb` and `notebooks/submissions/*.ipynb` succeeds; `data_module(path, 0)` builds for every `configs/models/exp*/*.yaml`; `CVResult.load(...).fold_model(0)` loads for `exp26_coarse_dropout_5fold` and `exp34_convnext_tiny`. Same script reused after every group.

## 1. Boxes

- [x] 1.1 Delete `src/core/boxes/`; drop "Box recipes" line from `src/core/data/augment/__init__.py`; verify `grep -rn "core.boxes\|from \.\.boxes" src tests` empty and baseline script passes.
- [x] 1.2 Remove `boxes/` line from `.claude/context/project-structure.md`; in `docs/methods/gradcam_augmentations.md` mark `src/core/boxes/pseudo.py` as removed (git history); verify `grep -rn "core/boxes\|core.boxes" docs .claude CLAUDE.md` shows only that marked line.

## 2. Model

- [x] 2.1 Delete `alexnet_imagenet`, `ConvBlock`, `ResnetConv`, `ResnetConvBlock`, `ResnetFromPaper` from `src/core/training/model.py`, imports only they used; update module docstring net list; keep legacy branch of `LitCNNMultiLabel.load`. Delete `tests/test_custom_model.py`. Verify pytest result same as 0.1 and baseline script (exp26 fold model load) passes.

## 3. Augmentations and dataset

- [x] 3.1 Delete recipes `Lecture`, `V2`, `V3NoCrop`, `GridDropout`, `Jpeg`, `JpegNonVip`, `GrayNonVip` from `recipes.py`; fix `augment/__init__.py` exports (`V2` gone). Verify `Default`, `Validation`, `CoarseDropoutNonVip` compose and run on random 224 px image.
- [x] 3.2 In `ops.py` keep only `NonVipDropout` (empty-holes override kept: already skips VIP rows, one method); drop `Transform`, `NonVip`, `NonVipImageCompression`, `NonVipToGray`. Verify on dummy image: `vip=True` output unchanged by holes, `vip=False` gets holes.
- [x] 3.3 Delete `augment/rows.py`; `MultilabelImageDataset.__getitem__` computes `vip`, calls `transform(image=image, vip=vip)["image"]`. Update `augment/__init__.py` docstring. Verify `data_module(CoarseDropoutNonVip config, 0)` yields one batch of right shape with `num_workers=2` (pickling).
- [x] 3.4 Drop `n_other` from `read_data` and `KaggleDataModule` (docstrings too); `read_data` merges into `read_train` callers if then identical. Verify baseline script passes.

## 4. Ensemble and analysis

- [x] 4.1 Delete `Stacking` from `src/core/training/ensemble.py`, its imports, docstring mention. Verify `MeanEnsemble(...)` of exp38 configs gives `oof_probs` without error.
- [x] 4.2 Drop `edits` / `edit` from `GradCamFigures.load`, `probs`, `image_and_cam`, `compare`. Verify `GradCamFigures(exp34 config).positives("Moris", n=2)` returns figure.
- [x] 4.3 Delete `preview`, `to_image`, `draw_boxes` from `src/core/analysis/plots.py`, orphaned imports; update `CLAUDE.md` helper line. Verify `grep -rn "preview\|to_image\|draw_boxes" src notebooks/experiments notebooks/submissions CLAUDE.md` empty.

## 5. Integration

- [x] 5.1 Rerun baseline script and pytest; `grep` notebooks for every removed name (`Stacking`, `n_other`, `edits`, removed recipes, removed nets) empty; pytest result same as 0.1; `.venv/bin/python -m core.training.cli fit --help` runs. No training, no notebook runs.

## 6. Tests and metrics

- [x] 6.1 Before edits, save `best_f1`, `plateau_f1` (margin 0.05), `cross_fold_f1` (best, margin 0.05) of exp38 `MeanEnsemble` OOF probs to scratch. Delete `f1_at` and `scores_at` from `src/core/analysis/metrics.py` (module docstring too); `plateau_f1` F1 via `sklearn.metrics.f1_score(..., zero_division=0)`; drop `scores_at` import in `thresholds.py`. Verify saved numbers equal after edit and `grep -rn "f1_at\|scores_at" src notebooks/experiments/*.ipynb notebooks/submissions/*.ipynb` empty.
- [x] 6.2 Delete `tests/` (`test_metrics.py`, `test_runs.py`); remove `tests/` line from `.claude/context/project-structure.md`. Verify baseline script passes and `grep -rn "unittest\|tests/" src CLAUDE.md .claude/context` empty.

## 7. Flatten augment

- [x] 7.1 Merge `src/core/data/augment/recipes.py` and `ops.py` into `src/core/data/augment.py` (`NonVipDropout` first, then `Recipe` tree; one module docstring); delete `src/core/data/augment/`. `dataset.py` imports `Default, Recipe, Validation` from `.augment`; `gradcam.py`, `submission.py` imports unchanged. Verify `Default`, `Validation`, `CoarseDropoutNonVip` compose on random 224 px image, `NonVipDropout` leaves `vip=True` image unchanged.
- [x] 7.2 Set `class_path: core.data.augment.CoarseDropoutNonVip` in exp26, exp32, exp33, exp34 configs (comment keeps only "colored holes on non-VIP rows"); `project-structure.md` `data/` line names `augment` module; `docs/methods/augmentations.md:19` points at `core.data.augment` and YAML `data.augment` `class_path` instead of `RECIPES` / `DataConfig`. Verify baseline script passes (data module of every config), `grep -rn "augment\.recipes\|augment\.ops\|augment/" src configs/models/exp* .claude/context docs/methods CLAUDE.md` empty, one batch of exp34 data module loads with `num_workers=2`.

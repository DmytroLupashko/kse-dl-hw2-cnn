# Proposal

## Why

`src/core/` grew to serve every experiment ever run, boxes and erase probes included. Main now holds only exp04–exp38; code no one on main uses makes core harder to read and to extend in later experiment branches.

## What Changes

- Delete `src/core/boxes/` (pseudo-boxes, exp06–exp24, never imported).
- Delete unused nets in `training/model.py`: `alexnet_imagenet`, `ConvBlock`, `ResnetConv`, `ResnetConvBlock`, `ResnetFromPaper`; delete `tests/test_custom_model.py`. Own ResNet lives on branch `resnet_from_paper`.
- Keep only recipes `Validation`, `Default`, `CoarseDropoutNonVip`; delete `Lecture`, `V2`, `V3NoCrop`, `GridDropout`, `Jpeg`, `JpegNonVip`, `GrayNonVip`.
- `augment/ops.py`: keep `NonVipDropout` only; drop `Transform` mixin, `NonVip`, `NonVipImageCompression`, `NonVipToGray`.
- Drop `RowTransform` and `row_extras` support; dataset passes `vip` flag to transform directly.
- Drop `n_other` from `read_data` and `KaggleDataModule`.
- Delete `Stacking` from `training/ensemble.py`.
- `GradCamFigures`: drop `edits` / `edit` (erase probes).
- `analysis/plots.py`: delete `preview`, `to_image`, `draw_boxes`.
- Delete `tests/test_metrics.py` and `tests/test_runs.py` (not maintained by owner), so whole `tests/` goes; delete `f1_at` from `analysis/metrics.py` (its only user was `test_metrics.py`).
- Delete `scores_at` from `analysis/metrics.py`; `plateau_f1` takes F1 at its middle threshold from `sklearn.metrics.f1_score`. `metrics.py` keeps `ap`, `pr_curve`; `thresholds.py` stays separate.
- Flatten `data/augment/` package into one module `data/augment.py` (`Recipe` tree plus `NonVipDropout`). **BREAKING** for 4 configs (exp26, exp32, exp33, exp34): `class_path` becomes `core.data.augment.CoarseDropoutNonVip`.
- Update docs naming removed code: `CLAUDE.md` helper line, `.claude/context/project-structure.md` (`boxes/`, `tests/` lines), `docs/methods/gradcam_augmentations.md` path mention, `docs/methods/augmentations.md` stale recipe pointer (line 19).

Out of scope: moving or renaming surviving modules other than `data/augment/`, flattening `data/` further (single `core/data.py`), merging `metrics.py` with `thresholds.py`, deleting old experiments from main, rerunning anything.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. Pure refactor, `skip_specs: true`.

## Impact

Code: `src/core/` (boxes, training/model, training/ensemble, data/dataset, data/augment, analysis/gradcam, analysis/plots, analysis/metrics), `tests/` (all three files). Docs: `CLAUDE.md`, `.claude/context/project-structure.md`, `docs/methods/gradcam_augmentations.md`, `docs/methods/augmentations.md`. Configs: `class_path` line of 4 model configs. Notebooks, run folders, `mlflow.db`: untouched, must keep importing and loading.

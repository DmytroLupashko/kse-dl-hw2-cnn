# Design

## Context

`positives` and `compare` both build 8-col grid: original row, then one overlay row per model, axis off, suptitle, `tight_layout`. Same figsize per row (2.4). Differences: `positives` has `train` switch (next fold), fold in label, config name in suptitle.

## Goals / Non-Goals

**Goals:** one grid builder for positives-style views; same public calls.

**Non-Goals:** merging `plot` (TP/FP/FN columns, 2 rows, true/pred titles; different layout); removing boxes-only paths; `GradCam` class to function.

## Decisions

- Extend `compare`, not new private `_grid`. `compare` already general (N models); `positives` = N=1. Alternative: callback-based `_grid(rows, draw)`; rejected, extra indirection for two callers.
- Fold choice inside `compare`: `fold = oof.loc[idx, "fold"]`, with `train`: next in `folds` of that model. Each `GradCamFigures` own folds list, so rule applied per model.
- Label `<name> · fold <k> · p <prob>` everywhere (user choice), no label parameter.
- Split text in suptitle from `train`: "train (model trained on image)" / "validation (model held image out)", reused from current `positives`.

## Risks / Trade-offs

- [Longer label at fontsize 8, 2.2" columns may clip] → not checked now; shows on next notebook rerun, shorten name key then.
- [No unit tests for gradcam; no figure check] → only `py_compile` and signature verified; pixel parity with old figures unverified until next rerun.

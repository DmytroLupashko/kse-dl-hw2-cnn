# Design

## Context

Finished runs live in `data/processed/cv/<experiment>/<name>/fold_<k>/`: checkpoint, `config.yaml`, `oof.parquet`, `run.json`. Notebooks read them through `CVResult.load`, `fold_model`, `GradCamFigures`, `Single`, `MeanEnsemble`. Checkpoints and run configs store dotted class paths. exp26 run (`cnn-model-coarse-dropout-5fold`) trained under older Hydra-style config: `_target_` blocks, `core.augment.recipes.CoarseDropoutNonVip`.

## Goals / Non-Goals

**Goals:**
- Every surviving notebook import and every config `class_path` resolves after change.
- Every finished run on main (exp04–exp34 run folders) still loads: OOF, checkpoint, Grad-CAM.

**Non-Goals:**
- Same RNG draws for future training runs.
- Moving, renaming or splitting surviving modules.

## Decisions

Delete, never move. Surviving names keep module paths (`core.training.model.timm_net`, `core.data.augment.recipes.CoarseDropoutNonVip`, `core.training.ensemble.MeanEnsemble`, ...): checkpoints, run configs, notebooks reference them. Alternative "reorganize while cutting" rejected: breaks loading for no reader benefit.

Own nets (`ResnetFromPaper` and its blocks) deleted from main: assignment work continues on branch `resnet_from_paper`. `model.py` keeps Lightning wrapper and `timm_net` only.

Keep legacy branch of `LitCNNMultiLabel.load` (`model` hparam, `_target_`, `core.` remap). exp26 checkpoints need it, exp26 notebook and exp38 ensemble read them. Removal would cost exp26 Grad-CAM.

`vip` flag goes straight from dataset to transform: `MultilabelImageDataset.__getitem__` computes it, calls `transform(image=..., vip=...)`. `RowTransform` existed for `row_extras` (box Compose); with boxes gone, class adds nothing. Alternative "keep `RowTransform`, drop extras" rejected: wrapper with one field.

`NonVipDropout` keeps empty-holes override for VIP rows: one method, skip covers every target, RNG draws unchanged. Plain `apply` skip rejected during apply: needs two overrides. `Transform.get_class_fullname` goes: recipe hash it served no longer exists.

`Default` stays: exp04/09/16/29 configs have no `augment`, so `KaggleDataModule` default is `Default`.

`data/augment/` flattens to `data/augment.py`, second exception to "delete, never move". Checked: exp26 and exp34 checkpoints pickle no `core.` path; run `config.yaml` files name older `core.augment.recipes...` and are never instantiated. Only 4 model configs change. One level only: `dataset.py` (sampling, datamodule) and `augment.py` (transforms) change for different reasons, so single `core/data.py` rejected.

No tests left on main. `test_metrics.py`, `test_runs.py` not maintained by owner; run folder failures loud, every `CVResult.load` exercises them. Proof for this and later changes: baseline script (notebook imports, data modules, fold model loads) plus before/after equality of numbers on real OOF.

`plateau_f1` F1 via `f1_score(y, p >= mid, zero_division=0)`: same value as `scores_at` (`2 TP / (predicted + positives)`), one helper less. Merge of `metrics.py` into `thresholds.py` rejected: `ap` is no threshold rule, 5 notebooks import `core.analysis.thresholds`.

`GradCamFigures.compare` stays without `edits`: exp26 notebook uses it, later experiments compare candidate vs baseline with it.

## Risks / Trade-offs

[Run config of exp26 names `core.augment.recipes.CoarseDropoutNonVip`, not current path] → nothing instantiates recipe from run config (only `img_size` read); unchanged by this change.

[Hidden user of removed name in a notebook cell not caught by import grep] → verification imports every notebook's `core` import lines and greps notebooks for removed names.

[Old checkpoints pickle something from removed module] → verification loads one fold model per run family: exp26 (legacy) and exp34 (current).

## Migration Plan

User does git. Rollback: revert commit.

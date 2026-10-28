# Card examples

Bad then good per field. Source: user critique of exp01 card, `drafts/verbosity.md`.

General: each field holds only its own kind of fact. Known context (validation mode, folds, loss, default config) never repeated. One item per line.

## Question

No setup inside: validation mode, folds, metric already known.

Bad:

```md
Which of 3 ImageNet-pretrained lecture 2 models (ResNet-18, EfficientNet-B0, AlexNet) gives highest mean per-fold mAP in Fast validation mode (folds 0 and 1 of 5-fold split), and how does per-class AP differ between them?
```

Good:

```md
How do ImageNet-pretrained ResNet-18, EfficientNet-B0, AlexNet differ?
```

## Plan

Steps name outputs only. No setup (folds, loss), no general policy (one cell per model).

Bad:

```md
1. Train each model on folds 0 and 1 with weighted BCE, one notebook cell per model, log `train_loss`, `val_loss`, `val_map` per epoch, plot mean over 2 folds per epoch, one line per model.
```

Good:

```md
1. Plot `train_loss`, `val_loss`, `val_map` per epoch, one line per model.
```

## Notebook

Notebook link only. No configs, no training module.

Bad:

```md
- **Notebook:** [exp01_lecture_models.ipynb](../notebooks/experiments/exp01_lecture_models.ipynb), configs `configs/models/exp01_lecture_models/`, training `core.training.cv`
```

Good:

```md
- **Notebook:** [exp01_lecture_models.ipynb](../notebooks/experiments/exp01_lecture_models.ipynb)
```

## Tracking

Experiment name only. No run layout, metric names, params.

Bad:

```md
- **Tracking:** `cnn-model-lecture-models` (one run per model and fold with `best_ap_<class>` metrics, run `summary` with param `val_mode` = `fast`)
```

Good:

```md
- **Tracking:** `cnn-model-lecture-models`
```

## What changes

Changed variable and its values, one per line. No dropped candidates, no past failures, no run ids.

Bad:

```md
- **What changes:** model, all ImageNet-1k pretrained (`docs/task/rules.md`): timm `resnet18.a1_in1k`, `tf_efficientnet_b0.in1k`, torchvision AlexNet `AlexNet_Weights.IMAGENET1K_V1` (no timm version, config field `torchvision_model`); scratch models dropped; ConvNeXt-Tiny (`convnext_tiny.fb_in1k`) dropped: too long to train; earlier AlexNet run collapsed at lr 1e-3, `val_map` 0.167 (constant output) all 10 epochs on both folds, MLflow runs `655e7053`, `f179a3d7`
```

Good:

```md
- **What changes:** model, ImageNet-1k pretrained:
  - `resnet18.a1_in1k`
  - `tf_efficientnet_b0.in1k`
  - torchvision `AlexNet_Weights.IMAGENET1K_V1`
```

## What stays fixed

One item per line. Defaults from `docs/methods/configuration.md`: link, not restated. No code paths.

Bad:

```md
- **What stays fixed:** lecture recipe: Adam, cosine lr 1e-3 to 1e-5, 10 epochs, batch 64, 224 px, `core.data.dataset.train_tfms`, mean = std = 0.5; config from `docs/methods/configuration.md`: `BCEWithLogitsLoss` with `pos_weight` $w_c = ((N - n_c)/n_c)^{3/4}$ computed on train part of fold, best checkpoint by `val_map` (added to `core.training.model`, `core.training.train`), folds from `core.data.dataset.read_train`, each fold model trains on other 4 folds (80%), `SEED`
```

Good:

```md
- **What stays fixed:**
  - lecture recipe: Adam, cosine lr 1e-3 to 1e-5, 10 epochs, batch 64, 224 px
  - loss, checkpoint, folds: `docs/methods/configuration.md`
```

## Metric

Metric and validation mode name only. Decision rules, display choices: not metric.

Bad:

```md
- **Metric:** per-class AP and mAP on validation fold, mean over folds 0 and 1; model below leader on both folds counts as worse; PR curves on pooled folds 0 and 1 only for display
```

Good:

```md
- **Metric:** per-class AP, mAP, Fast validation
```

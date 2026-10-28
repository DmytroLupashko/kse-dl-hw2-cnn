# Verbocity examples

## Examples

```md
## Question
Which of 3 ImageNet-pretrained lecture 2 models (ResNet-18, EfficientNet-B0, AlexNet) gives highest mean per-fold mAP in Fast validation mode (folds 0 and 1 of 5-fold split), and how does per-class AP differ between them?
```

Here should be: How ImageNet-pretrained ResNet-18, EfficientNet-B0, AlexNet differ?

Fast validation mode (folds 0 and 1 of 5-fold split)
We already know that this is fast validation mode, so we should not specify "folds 0 and 1 of 5 fold-split"

```md
## Plan
1. Train each model on folds 0 and 1 with weighted BCE, one notebook cell per model, log `train_loss`, `val_loss`, `val_map` per epoch, plot mean over 2 folds per epoch, one line per model.
2. Plot precision-recall curve of each class on pooled probabilities of folds 0 and 1, all 3 models on one figure, one figure per class.
3. Tabulate AP of each class plus mAP per model, mean over folds 0 and 1.
4. Tabulate mAP per fold and difference to leader per fold, flag models below leader on both folds.
```

Here, for example, you should not mention that model is trained on folds 0 and folds 1 with weighted BCE, because it is setup.
Also "one notebook cell per model" is redundant, as it is more like general policy then part of plan.

```md
- **Notebook:** [exp01_lecture_models.ipynb](../notebooks/experiments/exp01_lecture_models.ipynb), configs `configs/models/exp01_lecture_models/`, training `core.training.cv`
```

why configs included here, there should be only notebook.

```md
- **Tracking:** `cnn-model-lecture-models` (one run per model and fold with `best_ap_<class>` metrics, run `summary` with param `val_mode` = `fast`)
```

I think val mode should be defined as separate parameter; there should be just `cnn-model-lecture-models`

```md
- **What changes:** model, all ImageNet-1k pretrained (`docs/task/rules.md`): timm `resnet18.a1_in1k`, `tf_efficientnet_b0.in1k`, torchvision AlexNet `AlexNet_Weights.IMAGENET1K_V1` (no timm version, config field `torchvision_model`); scratch models dropped; ConvNeXt-Tiny (`convnext_tiny.fb_in1k`) dropped: too long to train; earlier AlexNet run collapsed at lr 1e-3, `val_map` 0.167 (constant output) all 10 epochs on both folds, MLflow runs `655e7053`, `f179a3d7`
```

WHAT A FUCK IS THAT? How I supposed to read this shit?

```md
- **What stays fixed:** lecture recipe: Adam, cosine lr 1e-3 to 1e-5, 10 epochs, batch 64, 224 px, `core.data.dataset.train_tfms`, mean = std = 0.5; config from `docs/methods/configuration.md`: `BCEWithLogitsLoss` with `pos_weight` $w_c = ((N - n_c)/n_c)^{3/4}$ computed on train part of fold, best checkpoint by `val_map` (added to `core.training.model`, `core.training.train`), folds from `core.data.dataset.read_train`, each fold model trains on other 4 folds (80%), `SEED`
```

At least, this should be separated by lines.

```md
- **Metric:** per-class AP and mAP on validation fold, mean over folds 0 and 1; model below leader on both folds counts as worse; PR curves on pooled folds 0 and 1 only for display
```

instead of "per-class AP and mAP on validation fold mean over folds 0 and 1" it could say "Fast validation"; how "model below leader on both folds counts as worse; PR curves on pooled folds 0 and 1 only for display" related to metric?

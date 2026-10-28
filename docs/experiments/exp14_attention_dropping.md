# Experiment 14: Attention dropping

## Request

create experiment for attention dropping

## Question

Does ResNet-18 with exp11 setup plus attention dropping (Grad-CAM pseudo-box erased, label kept) give higher mean per-fold mAP and Macro F1 on folds 0 and 1 than exp11 on same folds (mAP 0.967, Macro F1 0.952)?

## Plan

1. Plot 6 random draws of `attention_drop` on one training image per VIP class and one Cat-only image, one row per image, original first, pseudo-boxes drawn on each.
2. Train ResNet-18 on folds 0 and 1 with `attention_drop`, plot `train_loss`, `val_loss`, `val_map` per epoch, mean over 2 folds, one line per augmentation: default (exp11), `attention_drop`.
3. Tabulate AP of each class, mAP and Macro F1 (best-F1 thresholds tuned on same fold rows) per fold and mean over folds 0 and 1, for exp11 and `attention_drop`, plus per-fold difference to exp11, flag difference with same sign on both folds.
4. Plot Grad-CAM of exp11 and new model side by side on 3 case images of `docs/gradcam_augmentations.md` (hot spot beside cat 779, on cat 9101, on rug 2470), one row per labeled VIP class, fold 0 and fold 1 model of each, title says whether model held image out or trained on it.
5. Plot exp02 Grad-CAM figures (`core.gradcam.GradCamFigures`) for new model: per VIP class one "correct" and one "incorrect" figure, 6 figures.
6. Plot exp12 Grad-CAM figures (`GradCamFigures.positives`) for new model: per VIP class one validation figure (all labeled positives of folds 0 and 1: 22 Moris, 17 Motya, 16 Biatrix, fold model that held image out) and one train figure (same images, other fold model, which trained on image), 6 figures.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp14_attention_dropping.ipynb](../notebooks/experiments/exp14_attention_dropping.ipynb), config `configs/exp14_attention_dropping/resnet18.toml`, training `core.cv`
- **Tracking:** `cnn-model-attention-dropping`
- **What changes:** `augment = "attention_drop"`, new `core.data.attention_drop_tfms`: default recipe (`train_tfms`) with pseudo-box erase in front.
  - Erase: each pseudo-box of image (`data/processed/boxes.csv`, exp04 ResNet-18, map >= 0.5 of max) filled with zeros independently, p 0.5; labels kept.
  - Skip box with area > 0.3 of image: proxy for box covering whole animal (case 2), erasing it would leave label on image without animal. Median box area: Cat 0.15, Dog 0.16, Moris 0.26, Motya 0.04, Biatrix 0.10; 75th percentile Moris 0.33, Motya 0.29.
  - Applied to every image, not only VIP: VIP-only distortion teaches shortcut "erased patch means VIP".
  - Offline boxes, not WS-DAN online map: no extra backward pass, map fixed at exp04 model.
- **What stays fixed:** exp11 setup: `resnet18.a1_in1k`, `oversample = 8`, `pos_weight_power = 1.0`, Adam, cosine lr 1e-3 to 1e-5, 10 epochs, batch 64, 224 px, best checkpoint by `val_map`, `SEED`; exp11 folds 0 and 1 reused, not retrained. Validation `val_tfms`, never erased.
- **Metric:** per-class AP, mAP and Macro F1 per validation fold, mean over folds 0 and 1; Grad-CAM position, visual.

## Fixes

## Results

## AI Summary

## Conclusion

Applied as now these boxes close the whole animals.
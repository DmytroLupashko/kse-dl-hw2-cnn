# Experiment 33: RegNetY-016

## Request

create experiment cards for training effnet, regnety_016.tv2_in1k, convnext_tiny.fb_in1k

## Question

On all labeled VIP positives of folds 0–4, does RegNetY-016 with exp26 setup put Grad-CAM heat on body (not only face) in more images than exp26 ResNet-18, without lower VIP AP on pooled out-of-fold rows of folds 0–4?

## Plan

1. Plot precision-recall curve per class on pooled out-of-fold rows of folds 0–4, one panel per class (5), one line per model: exp26, RegNetY-016, AP in legend.
2. Plot Grad-CAM VIP validation figures (`GradCamFigures.positives`) of RegNetY-016: per VIP class all labeled positives of folds 0–4 (56 Moris, 42 Motya, 41 Biatrix), 3 figures, same layout as exp26.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp33_regnety_016.ipynb](../notebooks/experiments/exp33_regnety_016.ipynb), config `configs/exp33_regnety_016/regnety_016.toml`, training `core.cv`
- **Tracking:** `cnn-model-regnety-016`
- **What changes:** `timm_model = "regnety_016.tv2_in1k"` instead of `resnet18.a1_in1k`.
  - Grad-CAM layer: `s4` (last stage, 7 × 7 at 224 px), added to `last_feature_layer`.
- **What stays fixed:** exp26 setup (`augment = "coarse_dropout_non_vip"`, `oversample = 8`, `downsample = 0.2`, full ratio `pos_weight`), Adam lr 1e-3 to 1e-5, 10 epochs, 224 px, folds 0–4; exp26 results reused.
- **Metric:** count of VIP positives with Grad-CAM heat on body, by eye, vs exp26 figures (main); AP per VIP class on pooled out-of-fold rows of folds 0–4, from PR curves (guard, must not drop).

## Fixes

## Results

## AI Summary

## Conclusion

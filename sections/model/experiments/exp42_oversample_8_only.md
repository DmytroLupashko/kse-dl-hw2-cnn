# Experiment 42: Oversample 8 without downsampling

## Request

create an experiment: Oversample(8, 1) model with only initial augmentations; train it, compare precision recall curves for it with exp16; draw train/val curves for both folds

## Question

On pooled out-of-fold rows of folds 0–1, how do per-class precision-recall curves of ResNet-18 with `Oversample(8, 1)` differ from exp16 (`OverDownsample(8, 0.2)`)?

## Plan

1. Plot precision against recall per class for new model and exp16 on pooled out-of-fold rows of folds 0–1.
2. Plot train and val loss plus mAP per epoch for folds 0 and 1 of both models.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp42_oversample_8_only.ipynb](../notebooks/experiments/exp42_oversample_8_only.ipynb)
- **Tracking:** `cnn-model-oversample-8-only`; exp16 read from saved results
- **What changes:** `downsample` 1.0 (exp16: 0.2), so sampling `Oversample`, `pos_weight` power 0.5; `is_new = true`.
- **What stays fixed:** ResNet-18 `resnet18.a1_in1k`, `default` augmentations, `oversample` 8, folds 0–1, 10 epochs, 224 px, batch 64.
- **Metric:** AP per class, PR curves.

## Fixes

## Results

## AI Summary

## Conclusion

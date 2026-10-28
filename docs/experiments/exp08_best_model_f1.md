# Experiment 08: Best model Macro F1

## Request

create experiment 8, show current macro F1 and precision recall curves for current best model

## Question

What Macro F1 does current best model (exp04 ResNet-18, mAP 0.972) reach on out-of-fold rows of folds 0 and 1 with per-class thresholds, when thresholds are tuned on same rows and when tuned on other fold?

## Plan

1. Tabulate threshold, precision, recall, F1 per class and Macro F1 of exp04 on out-of-fold rows of folds 0 and 1, two threshold modes: tuned on same rows (optimistic), tuned on other fold (fold 0 thresholds scored on fold 1, and back).
2. Plot precision-recall curve per class on out-of-fold rows of folds 0 and 1, one panel per class, best-F1 point marked.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp08_best_model_f1.ipynb](../notebooks/experiments/exp08_best_model_f1.ipynb), no training
- **Tracking:** none
- **What changes:** nothing trained; evaluation only. Thresholds: `core.predict.best_f1` (max F1 per class), same rule as `docs/configuration.md`.
- **What stays fixed:** exp04 results `data/processed/cv/cnn-model-oversampling-vip/ResNet-18/` loaded via `CVResult.load`, config `configs/exp04_oversampling_vip/resnet18.toml`.
- **Metric:** Macro F1 (target), mean of 5 per-class F1, scored on out-of-fold rows of folds 0 and 1

## Fixes

## Results

## AI Summary

## Conclusion

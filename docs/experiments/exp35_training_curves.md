# Experiment 35: Training curves of submissions

## Request

for previous successful submissions (exp11, exp26, exp29) add training / validation curves

## Question

On folds 0–4, how do train and validation loss and mAP change per epoch for exp11, exp26 and exp29, and at which epoch does validation stop improving?

## Plan

1. Plot train and validation loss per epoch, one panel per model (exp11, exp26, exp29), one thin line per fold, train and validation in two colors.
2. Plot train and validation mAP per epoch, same layout.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp35_training_curves.ipynb](../notebooks/experiments/exp35_training_curves.ipynb)
- **Tracking:** none; reads fold runs of `cnn-model-resnet18-5fold`, `cnn-model-coarse-dropout-5fold`, `cnn-model-downsample-5fold`
- **What changes:** nothing trained; per-epoch `train_loss_epoch`, `val_loss`, `train_map`, `val_map` from MLflow history.
- **What stays fixed:** logged runs as is, folds 0–4.
- **Metric:** loss as logged and mAP per epoch, train vs validation fold rows.

## Fixes

## Results

## AI Summary

## Conclusion

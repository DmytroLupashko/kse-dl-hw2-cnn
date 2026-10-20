# Backlog

Source: `docs/Assignment_2_Task.md`. Task context: `docs/overview.md`.

- [x] Set up workflow (Github + local structure)
- [ ] Code is clean and readable
- [ ] Write README.md

## EDA and Metric Analysis

Section: `sections/eda/`.

- [x] EDA: class balance, label co-occurrence, labels per image, image sizes, sample grids
- [x] Macro F1 analysis: strengths, weaknesses, edge cases (rare class, empty predictions, threshold sensitivity)
- [x] Complementary metrics: per-class F1, micro F1, mAP, ROC AUC
- [x] Alternative target metric, justified
## Data Augmentation

Section: `sections/model/`.
## Validation

Section: `sections/validation/`.

- [x] Validation scheme (multilabel stratified K-fold or holdout) based on common sense
- [x] Motivation for chosen scheme
- [ ] Reasons for low or high LB correlation

## CNN Model

Section: `sections/model/`. Tracking: MLflow, `cnn-model-<slug>`.

- [ ] Experiment + logging infra (PyTorch Lightning, MLflow)
- [ ] Baseline timm CNN
- [ ] Sweeps: optimizers, schedulers, LR, augmentations, loss, thresholds, other lecture topics
- [ ] Own CNN or own block in existing architecture; compare with timm baseline
- [ ] Collect results:
  - [ ] Loss/metric curves
  - [ ] Gradient flow plots
  - [ ] GradCAM
  - [ ] Tuning logs
- [ ] Submit; compare Public LB vs local validation
- [ ] Verdict: model adequate or not

## Ensembling

Section: `sections/ensemble/`.

- [ ] Ensemble several CNNs (averaging, voting; not stacking)
- [ ] [Optional] Logistic Regression stacking
- [ ] Submit; compare Public LB vs local validation



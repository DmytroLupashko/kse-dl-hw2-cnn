# Backlog

Source: `docs/Assignment_2_Task.md`. Task context: `docs/overview.md`.

- [ ] Set up workflow (Github + local structure)
- [ ] Code is clean and readable
- [ ] Write README.md

## EDA and Metric Analysis

Section: `sections/eda/`.

- [ ] EDA: class balance, label co-occurrence, labels per image, image sizes, sample grids
- [ ] Macro F1 analysis: strengths, weaknesses, edge cases (rare class, empty predictions, threshold sensitivity)
- [ ] Complementary metrics: per-class F1, micro F1, mAP, ROC AUC
- [ ] Alternative target metric, justified
## Data Augmentation

Section: `sections/augmentation/`.
## Validation

Section: `sections/validation/`.

- [ ] Validation scheme (multilabel stratified K-fold or holdout), correlated with Public LB
- [ ] Motivation for chosen scheme
- [ ] Reasons for low or high LB correlation
- [ ] Adversarial validation if needed, same model as main task

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



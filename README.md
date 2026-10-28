# Homework 2: CNN

## Project structure

```text
├── src/
│   ├── core/       project code, one Python package
│   │   ├── data/       datasets, folds, augmentations
│   │   ├── training/   paths, configs, cross-validation, final fit, ensembles, submissions, model, tracking
│   │   ├── analysis/   scores, Grad-CAM, figures
│   │   └── boxes/      experimental Grad-CAM pseudo-boxes, unused later
├── configs/
│   └── models/       training configs, `expNN_<slug>/<model>.yaml`
├── notebooks/
│   ├── experiments/  one notebook per card
│   ├── submissions/  notebooks that write submission
│   └── playground/   scratch (not in git)
├── tests/
├── docs/
│   ├── task/         task, rules, assignment
│   ├── methods/      metrics, validation, augmentations, oversampling, ensembling
│   ├── reports/      eda, validation, model, ensemble reports
│   ├── experiments/  one card per experiment
│   ├── figures/      all figures
│   ├── backlog.md
│   └── course/       Module 2 course materials (not in git)
├── data/           not in git
│   ├── kaggle/       competition data as downloaded
│   ├── processed/    generated files
│   └── submissions/  CSVs for Kaggle
└── ai_usage/       AI chat transcripts
```

## AI usage

AI tools helped write parts of the code and the reports. Full transcripts are in [ai_usage/](ai_usage/).

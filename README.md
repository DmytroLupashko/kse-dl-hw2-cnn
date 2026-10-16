# Homework 2: CNN

## Project structure

```text
├── core/           project code, one Python package
├── sections/       experiments, in workflow order
│   ├── eda/
│   ├── validation/
│   ├── baseline/
│   ├── augmentation/
│   ├── model/
│   └── ensemble/
├── docs/           task, rules, backlog
│   └── course/     Module 2 course materials (not in git)
├── data/           not in git
│   ├── kaggle/       competition data as downloaded
│   ├── processed/    generated files
│   └── submissions/  CSVs for Kaggle
└── ai_usage/       AI chat transcripts
```

Every section folder has same layout, folders appear with first file:

```text
<section>/
├── <section>_report.md     section report
├── experiments/            one card per experiment
├── notebooks/
│   ├── experiments/        one notebook per card
│   └── submissions/        notebooks that write submission
└── figures/                figures used in report
```

## AI usage

AI tools helped write parts of the code and the reports. Full transcripts are in [ai_usage/](ai_usage/).

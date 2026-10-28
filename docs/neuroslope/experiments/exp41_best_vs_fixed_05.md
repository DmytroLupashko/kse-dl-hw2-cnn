# Experiment 41: Best vs fixed 0.5 thresholds

## Request

for exp16 model evaluate F1 score for best thresholds and for purely 0.5, show result as table

## Question

On pooled out-of-fold rows of folds 0–4, how do per-class F1 and Macro F1 of exp29 ResNet-18 (5-fold exp16) differ between best-F1 thresholds and all thresholds 0.5?

## Plan

1. Tabulate threshold and F1 per class plus Macro F1 for both rules, best-F1 and all 0.5, on pooled out-of-fold rows of exp29.
2. Plot precision against recall for Moris, Motya, Biatrix, with best-F1 and 0.5 points marked.

All plots follow `dataviz` skill.

## Setup

- **Notebook:** [exp41_best_vs_fixed_05.ipynb](../../notebooks/experiments/exp41_best_vs_fixed_05.ipynb)
- **Tracking:** none; reads saved results of exp29
- **What changes:** threshold rule: argmax F1 per class vs 0.5 for every class.
- **What stays fixed:** out-of-fold probs as saved, folds 0–4, prediction rule prob ≥ threshold.
- **Metric:** F1 per class and Macro F1.

## Fixes

cahnge the experiment: use 5 fold version of this model instead of exp16

Also, I want to see precision recall curves for this model

## Results

| class | best-F1 threshold | best-F1 F1 | all 0.5 F1 |
|---|---|---|---|
| Cat | 0.484 | 0.983 | 0.982 |
| Dog | 0.600 | 0.976 | 0.974 |
| Moris | 0.858 | 0.885 | 0.812 |
| Motya | 0.888 | 0.889 | 0.784 |
| Biatrix | 0.771 | 0.964 | 0.882 |
| **Macro F1** | | **0.939** | **0.887** |

F1 per class and Macro F1 on pooled out-of-fold rows of folds 0–4, exp29; best-F1 thresholds tuned on same rows (optimistic).

## AI Summary

Cat, Dog flat between rules; all three VIP classes clearly down at 0.5, so Macro F1 down. Best VIP thresholds sit high, well above 0.5, so 0.5 gives too many false positives there; part of gap is optimism of tuning on scored rows.

## Conclusion

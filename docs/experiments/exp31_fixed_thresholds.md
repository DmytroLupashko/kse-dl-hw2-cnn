# Experiment 31: Fixed thresholds

## Request

create new experiment, I want a table to compare current optimistic macro F1 vs macro F1 with thresholds 0.5 0.5 0.8 0.8 0.8; for hard coded thresholds also add TP / FP / FN table; You can even not create notebook, just do it inside card's results section

## Question

On out-of-fold rows of folds 0–4, do fixed thresholds 0.5 0.5 0.8 0.8 0.8 (Cat, Dog, Moris, Motya, Biatrix) give lower Macro F1 than best-F1 thresholds, for exp11, exp29 and exp26?

## Plan

1. Table: Macro F1 per model at best-F1, best VIP, 0.8 0.8 0.8, 0.9 0.9 0.7 and 0.85 0.9 0.75 thresholds.
2. Table: TP, FP, FN per VIP class and threshold setting, one table per model.

## Setup

- **Notebook:** [exp31_fixed_thresholds.ipynb](../notebooks/experiments/exp31_fixed_thresholds.ipynb), saved CV results (`CVResult.load`), hand thresholds in `THRESHOLDS`
- **Tracking:** none
- **What changes:** Cat, Dog 0.5; VIP thresholds best-F1, 0.8 0.8 0.8, 0.9 0.9 0.7, or 0.85 0.9 0.75 (Moris, Motya, Biatrix).
- **What stays fixed:** exp11, exp29, exp26 OOF predictions; best-F1 thresholds by `core.predict.best_f1` on pooled rows (submission thresholds).
- **Metric:** Macro F1 on pooled OOF rows of folds 0–4 (12260 rows).

## Fixes

cahnge TP / FP / FN table: make columns TP best / TP fixed

separate this table on 3 tables for each experiment separately

add to results tables for 0.5 0.5 (for all): 0.8 0.8 0.8; 0.9 0.9 0.7 (beatrix 0.7); best (only for VIPs). write summary about what you see

add to experiment card initial thresholds for each model

add also 0.9 Motya, 0.85 Moris, 0.75 Beatrix

## Results

| model | Cat | Dog | Moris | Motya | Biatrix |
|---|---|---|---|---|---|
| exp11 | 0.498 | 0.565 | 0.840 | 0.940 | 0.939 |
| exp29 | 0.484 | 0.600 | 0.858 | 0.888 | 0.771 |
| exp26 | 0.417 | 0.625 | 0.895 | 0.821 | 0.875 |

Initial thresholds: best-F1 per class on pooled OOF rows, used in Kaggle submissions.

Settings: Cat, Dog at 0.5 in all except "best all"; VIP (Moris, Motya, Biatrix) at best-F1, 0.8 0.8 0.8, 0.9 0.9 0.7, or 0.85 0.9 0.75.

| model | best all | best VIP | 0.8 0.8 0.8 | 0.9 0.9 0.7 | 0.85 0.9 0.75 |
|---|---|---|---|---|---|
| exp11 | 0.915 | 0.915 | 0.906 | 0.896 | 0.901 |
| exp29 | 0.939 | 0.939 | 0.922 | 0.923 | 0.933 |
| exp26 | 0.921 | 0.920 | 0.907 | 0.903 | 0.898 |

Macro F1. "best all": every class at own best-F1 threshold (optimistic, submission thresholds).

exp11:

| class | setting | TP | FP | FN |
|---|---|---|---|---|
| Moris | best | 48 | 6 | 8 |
| Moris | 0.8 0.8 0.8 | 48 | 7 | 8 |
| Moris | 0.9 0.9 0.7 | 44 | 5 | 12 |
| Moris | 0.85 0.9 0.75 | 46 | 6 | 10 |
| Motya | best | 33 | 7 | 9 |
| Motya | 0.8 0.8 0.8 | 33 | 10 | 9 |
| Motya | 0.9 0.9 0.7 | 33 | 9 | 9 |
| Motya | 0.85 0.9 0.75 | 33 | 9 | 9 |
| Biatrix | best | 38 | 2 | 3 |
| Biatrix | 0.8 0.8 0.8 | 39 | 4 | 2 |
| Biatrix | 0.9 0.9 0.7 | 39 | 7 | 2 |
| Biatrix | 0.85 0.9 0.75 | 39 | 6 | 2 |

exp29:

| class | setting | TP | FP | FN |
|---|---|---|---|---|
| Moris | best | 50 | 7 | 6 |
| Moris | 0.8 0.8 0.8 | 50 | 10 | 6 |
| Moris | 0.9 0.9 0.7 | 45 | 6 | 11 |
| Moris | 0.85 0.9 0.75 | 50 | 9 | 6 |
| Motya | best | 36 | 3 | 6 |
| Motya | 0.8 0.8 0.8 | 36 | 5 | 6 |
| Motya | 0.9 0.9 0.7 | 35 | 3 | 7 |
| Motya | 0.85 0.9 0.75 | 35 | 3 | 7 |
| Biatrix | best | 40 | 2 | 1 |
| Biatrix | 0.8 0.8 0.8 | 37 | 2 | 4 |
| Biatrix | 0.9 0.9 0.7 | 40 | 4 | 1 |
| Biatrix | 0.85 0.9 0.75 | 40 | 2 | 1 |

exp26:

| class | setting | TP | FP | FN |
|---|---|---|---|---|
| Moris | best | 49 | 7 | 7 |
| Moris | 0.8 0.8 0.8 | 50 | 15 | 6 |
| Moris | 0.9 0.9 0.7 | 48 | 7 | 8 |
| Moris | 0.85 0.9 0.75 | 49 | 11 | 7 |
| Motya | best | 34 | 8 | 8 |
| Motya | 0.8 0.8 0.8 | 34 | 9 | 8 |
| Motya | 0.9 0.9 0.7 | 29 | 5 | 13 |
| Motya | 0.85 0.9 0.75 | 29 | 5 | 13 |
| Biatrix | best | 40 | 2 | 1 |
| Biatrix | 0.8 0.8 0.8 | 40 | 3 | 1 |
| Biatrix | 0.9 0.9 0.7 | 41 | 6 | 0 |
| Biatrix | 0.85 0.9 0.75 | 40 | 5 | 1 |

Pooled out-of-fold rows of folds 0–4, 12260 rows.

## AI Summary

Yes: both hand settings lower Macro F1 for all three models; best VIP thresholds stay top, Cat, Dog at 0.5 cost nothing. 0.8 adds false positives without fewer misses; 0.9 0.9 0.7 trades Moris (exp11, exp29) and Motya (exp26) misses for Biatrix false positives, 0.85 0.9 0.75 closest to best VIP for exp29, not for exp11, exp26; exp29 top in every setting.

## Conclusion

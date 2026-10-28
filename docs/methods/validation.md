# Validation Strategies for Classification

Selection (compare models, thresholds, epochs) vs estimation (report chosen model's score); bias vs variance of estimate.

Read:

- Raschka (2018), [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808). Sections 1–4: holdout, bootstrap, k-fold, nested CV, selection bias. Best single overview for whole doc.
- Hastie, Tibshirani, Friedman, [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), ch. 7 "Model Assessment and Selection", 7.10 cross-validation.

## Holdout

Read: Raschka (2018), section 2: holdout, three-way split, variance of one split.

## K-fold cross-validation

Read:

- Kohavi (1995), [A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf), IJCAI. Stratified 10-fold vs leave-one-out vs bootstrap.
- scikit-learn, [3.1 Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). `KFold`, `RepeatedKFold`, `LeaveOneOut`, `cross_val_predict` (OOF predictions).

### Pooled OOF score vs mean of fold scores

Read: Forman, Scholz, [Apples-to-Apples in Cross-Validation Studies: Pitfalls in Classifier Performance Measurement](https://dl.acm.org/doi/10.1145/1882471.1882479), SIGKDD Explorations (2010). Pooled counts for F1 under imbalance; mean of fold values for AUC.

### When mean of fold scores is better

Project rule, from Forman and Scholz:

1. Thresholded metrics (F1) on rare classes: pool counts over folds. Pooled OOF array also used for per-class threshold tuning.
2. mAP as selection metric: AP per class inside each fold, mean over folds, then mean over classes. Pooled mAP diagnostic only.
3. Pooled AP clearly below mean of fold APs: fold models disagree on score scale; treat pooled threshold with care.

## Stratified k-fold

Read: Kohavi (1995), above: stratification lowers variance of CV estimate. `sklearn.model_selection.StratifiedKFold`, `train_test_split(..., stratify=y)`.

### Multilabel stratification

`StratifiedKFold` refuses multilabel matrix: "Supported target types are: ('binary', 'multiclass'). Got 'multilabel-indicator' instead."

Read:

- Sechidis, Tsoumakas, Vlahavas (2011), [On the Stratification of Multi-Label Data](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10), ECML PKDD. Iterative stratification; label-combination stratification breaks with many rare combinations.
- Szymański, Kajdanowicz (2017), [A Network Perspective on Stratification of Multi-Label Data](https://proceedings.mlr.press/v74/szyma%C5%84ski17a.html), PMLR 74. Second-order variant, balances label pairs.

Implementations: `iterstrat.ml_stratifiers.MultilabelStratifiedKFold` (package `iterative-stratification`), `skmultilearn.model_selection.IterativeStratification`.

This dataset, 12260 images, 5 labels, 5 folds. Per-fold count of positives, min to max over folds, worst of 3 seeds:

| Split | Cat | Dog | Moris (56) | Motya (42) | Biatrix (41) |
|---|---|---|---|---|---|
| `KFold`, random | 980-1046 | 974-1032 | 7-20 | 7-11 | 3-13 |
| `StratifiedKFold` on Cat | 1015-1016 | 982-1037 | 3-17 | 3-14 | 4-12 |
| `StratifiedKFold` on combination | 1014-1016 | 1004-1006 | 10-12 | 8-9 | 8-9 |
| `MultilabelStratifiedKFold` | 1015-1016 | 1005-1005 | 11-12 | 8-9 | 8-9 |

Combination stratification almost matches multilabel here: only 8 combinations, smallest 4 images.

## Group k-fold

Read: Kaufman, Rosset, Perlich, Stitelman (2012), [Leakage in Data Mining: Formulation, Detection, and Avoidance](https://dl.acm.org/doi/10.1145/2382577.2382579), ACM TKDD. `GroupKFold`, `StratifiedGroupKFold` in scikit-learn guide above.

Here: `Moris`, `Motya`, `Biatrix` are individual animals; photos from same session share background and lighting. Split by session, if recoverable, gives honest estimate of recognizing animal in new settings.

## Time-based split

Read: Hyndman, Athanasopoulos, [Forecasting: Principles and Practice, 5.10 Time series cross-validation](https://otexts.com/fpp3/tscv.html). `TimeSeriesSplit`. Not needed here unless collection drifted.

## Nested cross-validation

Read: Cawley, Talbot (2010), [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html), JMLR 11. Raschka (2018), section 4.

## Choosing strategy

Read:

- Dietterich (1998), [Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms](https://doi.org/10.1162/089976698300017197), Neural Computation. Paired comparison on same splits.
- Bouthillier et al. (2021), [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098), MLSys. Deep learning: seed and split variance vs difference between models.
- Kapoor, Narayanan (2023), [Leakage and the Reproducibility Crisis in Machine-Learning-Based Science](https://arxiv.org/abs/2207.07048), Patterns. Leakage taxonomy: preprocessing on full data, resampling before split, duplicates, selection on test.

## Imbalanced classes

Which metric under imbalance: `docs/metrics.md`.

### Noise from few positives

Read:

- Brown, Cai, DasGupta (2001), [Interval Estimation for a Binomial Proportion](https://doi.org/10.1214/ss/1009213286), Statistical Science. Recall on $n$ positives is binomial proportion; $SE(\hat r) = \sqrt{r(1-r)/n}$.
- Efron, Tibshirani (1993), *An Introduction to the Bootstrap*. Bootstrap confidence interval of metric without retraining.

Here: 41 positives, 5 folds, about 8 per fold, $SE \approx 0.14$ at $r = 0.8$, one image moves recall by 0.125. Cat and Dog, about 1000 per fold, $SE \approx 0.01$. Macro noise comes almost entirely from VIP classes; differences below ~0.05 Macro F1 per fold are noise.

### K-fold only for rare classes?

No paper; conclusion. Rotating only rare images through folds saves nothing: cost is number of trainings, $k$ either way. Iterative stratification on all labels balances rare and frequent at once.

### Threshold tuning is selection

Read: Cawley, Talbot (2010), above; Lipton et al. (2014), cited in `docs/metrics.md`. Tune per-class threshold on pooled OOF predictions; 8 positives give only 9 recall values, optimum jagged.

### Resampling and weighting interact with validation

Read:

- Santos et al. (2018), [Cross-Validation for Imbalanced Datasets: Avoiding Overoptimistic and Overfitting Approaches](https://doi.org/10.1109/MCI.2018.2866730), IEEE Computational Intelligence Magazine. Resample inside training part only.
- Guo et al. (2017), [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). Temperature scaling; Platt and isotonic as baselines.

### Hierarchical or correlated labels

Here: VIP class implies `Cat` or `Dog`. Count predictions violating it. Iterative stratification on all labels keeps structure; second-order variant (Szymański, Kajdanowicz) balances label pairs.

## Summary of decisions

1. Pick split unit: sample, or group if samples related.
2. Stratify by labels; iterative stratification for multilabel.
3. Holdout for iteration, k-fold for final comparison, pooled OOF for thresholds.
4. Estimate noise (fold spread, bootstrap) before trusting differences.
5. Keep test data out of every decision.

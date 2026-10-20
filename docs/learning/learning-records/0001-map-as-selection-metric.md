# mAP as selection metric: area as proxy for best F1

User explained in own words why mAP selects model when target is Macro F1: iso-F1 geometry turns "find best F1" into "find best set of PR curves", one per class, then tune $t_c$ per class. Understands AP as area under PR curve, F1 as one point on it, ROC vs PR under imbalance.

**Evidence:** own summary after `docs/metrics.md` sections on AP, iso-F1, ROC vs PR (2026-10-17); own EDA note "F1 is harmonic mean between precision and recall, so it is much natural to use mAP".

**Implications:** next gap: area vs peak can disagree (same AP, different best F1), and noise of per-class threshold tuning with ~40 positives. Good next topics: threshold tuning on small validation set, cross-validated thresholds.

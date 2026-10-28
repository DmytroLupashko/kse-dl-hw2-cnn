"""Metrics of probabilities against 0/1 labels. `y`, `probs`: arrays (rows, classes); `classes`: names, result index."""

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def ap(y, probs, classes):
    """AP of each class; `.mean()`: mAP (selection metric)."""
    return pd.Series(
        average_precision_score(y, probs, average=None), index=classes, name="ap"
    )


def pr_curve(y, p):
    """(precision, recall, thresholds, F1 per threshold) of one class (`y`, `p` 1-D); last curve point (recall 0) has
    no threshold, no F1."""
    precision, recall, thr = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision, recall, thr, f1[:-1]

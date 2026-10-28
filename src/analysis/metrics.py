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


def scores_at(y, p, threshold):
    """(precision, recall, F1) of one class (`y`, `p` 1-D) predicting 1 if prob >= `threshold`."""
    pred = p >= threshold
    positives = y.sum()
    tp = (pred & (y == 1)).sum()
    precision = tp / pred.sum() if pred.sum() else 0.0

    return precision, tp / positives, 2 * tp / (pred.sum() + positives)


def f1_at(y, probs, thresholds, classes):
    """Per class: given threshold (one per class), its precision, recall, F1."""
    rows = [(t, *scores_at(y[:, c], probs[:, c], t)) for c, t in enumerate(thresholds)]

    return pd.DataFrame(
        rows, index=classes, columns=["threshold", "precision", "recall", "f1"]
    )

"""Threshold per class from probabilities and 0/1 labels (predict 1 if prob >= threshold): best F1, plateau F1, and
cross-fold F1 (thresholds tuned on other folds). `y`, `probs`: arrays (rows, classes); `classes`: names, result index."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .metrics import pr_curve, scores_at


def best_f1(y, probs, classes):
    """Per class: threshold maximizing F1, its precision, recall, F1."""
    rows = []

    for c in range(len(classes)):
        precision, recall, thr, f1 = pr_curve(y[:, c], probs[:, c])
        best_idx = f1.argmax()
        rows.append(
            (thr[best_idx], precision[best_idx], recall[best_idx], f1[best_idx])
        )

    return pd.DataFrame(
        rows, index=classes, columns=["threshold", "precision", "recall", "f1"]
    )


def plateau_f1(y, probs, classes, margin=0.05):
    """Per class: middle of contiguous threshold range around best-F1 peak where F1 >= max - `margin` (`low`, `high`),
    its F1; less sensitive to probability shift than peak threshold."""
    rows = []

    for c in range(len(classes)):
        _, _, thr, f1 = pr_curve(y[:, c], probs[:, c])

        lo = hi = f1.argmax()

        while lo > 0 and f1[lo - 1] >= f1.max() - margin:
            lo -= 1

        while hi < len(f1) - 1 and f1[hi + 1] >= f1.max() - margin:
            hi += 1

        mid = (thr[lo] + thr[hi]) / 2
        rows.append((mid, thr[lo], thr[hi], scores_at(y[:, c], probs[:, c], mid)[2]))

    return pd.DataFrame(rows, index=classes, columns=["threshold", "low", "high", "f1"])


def cross_fold_f1(y, probs, folds, classes, margin=None):
    """F1 per class, each fold of row array `folds` thresholded by `best_f1` (`margin` None) or `plateau_f1(margin)`
    tuned on other folds; `.mean()`: honest Macro F1."""
    folds = np.asarray(folds)
    pred = np.zeros(probs.shape, int)

    for fold in np.unique(folds):
        rest, part = folds != fold, folds == fold

        if margin is None:
            tuned = best_f1(y[rest], probs[rest], classes)
        else:
            tuned = plateau_f1(y[rest], probs[rest], classes, margin)

        pred[part] = probs[part] >= tuned["threshold"].to_numpy()

    return pd.Series(
        f1_score(y, pred, average=None, zero_division=0), index=classes, name="f1"
    )

"""Scores of probabilities against 0/1 labels: per-class AP, best F1 and plateau F1 thresholds."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import CLASSES


class Scores:
    """`y_true`, `probs`: arrays (rows, classes), one column per class of `CLASSES`."""

    def __init__(self, y_true, probs):
        self.y_true = y_true
        self.probs = probs

    def ap(self):
        """AP of each class plus mAP (selection metric)."""
        per_class = average_precision_score(self.y_true, self.probs, average=None)
        return pd.Series(np.append(per_class, per_class.mean()), index=[*CLASSES, "mAP"], name="ap")

    def _f1_curve(self, c):
        """(precision, recall, thresholds, F1 per threshold) of class `c`; last curve point (recall 0) has no threshold."""
        precision, recall, thr = precision_recall_curve(self.y_true[:, c], self.probs[:, c])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return precision, recall, thr, f1[:-1]

    def best_f1(self):
        """Per class: threshold maximizing F1 (predict 1 if prob >= threshold), its precision, recall, F1."""
        rows = []
        for c in range(len(CLASSES)):
            precision, recall, thr, f1 = self._f1_curve(c)
            i = f1.argmax()
            rows.append((thr[i], precision[i], recall[i], f1[i]))
        return pd.DataFrame(rows, index=CLASSES, columns=["threshold", "precision", "recall", "f1"])

    def plateau_f1(self, margin=0.05):
        """Per class: middle of contiguous threshold range around best-F1 peak where F1 >= max - `margin` (`low`, `high`),
        its F1; less sensitive to probability shift than peak threshold."""
        rows = []
        for c in range(len(CLASSES)):
            _, _, thr, f1 = self._f1_curve(c)
            lo = hi = f1.argmax()
            while lo > 0 and f1[lo - 1] >= f1.max() - margin:
                lo -= 1
            while hi < len(f1) - 1 and f1[hi + 1] >= f1.max() - margin:
                hi += 1
            mid = (thr[lo] + thr[hi]) / 2
            pred = self.probs[:, c] >= mid
            tp = (pred & (self.y_true[:, c] == 1)).sum()
            rows.append((mid, thr[lo], thr[hi], 2 * tp / (pred.sum() + self.y_true[:, c].sum())))
        return pd.DataFrame(rows, index=CLASSES, columns=["threshold", "low", "high", "f1"])

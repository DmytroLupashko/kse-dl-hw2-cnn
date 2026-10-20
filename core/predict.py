"""Probabilities of a trained model, per-class AP, best F1 and plateau F1."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

from .config import CLASSES


@torch.no_grad()
def predict_probs(model, loader):
    """Sigmoid probabilities and labels of every row of `loader`, as numpy arrays."""
    model.eval()
    probs, targets = [], []
    for x, y in tqdm(loader, desc="predict"):
        probs.append(torch.sigmoid(model(x.to(model.device)).float()).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def ap_report(y_true, probs):
    """AP of each class plus mAP (selection metric)."""
    per_class = average_precision_score(y_true, probs, average=None)
    return pd.Series(np.append(per_class, per_class.mean()), index=[*CLASSES, "mAP"], name="ap")



def best_f1(y_true, probs):
    """Per class: threshold maximizing F1 (predict 1 if prob >= threshold), its precision, recall, F1."""
    rows = []
    for c in range(len(CLASSES)):
        precision, recall, thr = precision_recall_curve(y_true[:, c], probs[:, c])
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        i = f1[:-1].argmax()  # last curve point (recall 0) has no threshold
        rows.append((thr[i], precision[i], recall[i], f1[i]))
    return pd.DataFrame(rows, index=CLASSES, columns=["threshold", "precision", "recall", "f1"])


def plateau_f1(y_true, probs, margin=0.05):
    """Per class: middle of contiguous threshold range around best-F1 peak where F1 >= max - `margin` (`low`, `high`),
    its F1; less sensitive to probability shift than peak threshold."""
    rows = []
    for c in range(len(CLASSES)):
        precision, recall, thr = precision_recall_curve(y_true[:, c], probs[:, c])
        f1 = (2 * precision * recall / (precision + recall + 1e-12))[:-1]
        lo = hi = f1.argmax()
        while lo > 0 and f1[lo - 1] >= f1.max() - margin:
            lo -= 1
        while hi < len(f1) - 1 and f1[hi + 1] >= f1.max() - margin:
            hi += 1
        mid = (thr[lo] + thr[hi]) / 2
        pred = probs[:, c] >= mid
        tp = (pred & (y_true[:, c] == 1)).sum()
        rows.append((mid, thr[lo], thr[hi], 2 * tp / (pred.sum() + y_true[:, c].sum())))
    return pd.DataFrame(rows, index=CLASSES, columns=["threshold", "low", "high", "f1"])

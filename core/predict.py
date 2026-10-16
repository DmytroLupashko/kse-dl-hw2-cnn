"""Probabilities of a trained model and per-class F1 at a threshold."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import CLASSES


@torch.no_grad()
def predict_probs(model, loader):
    """Sigmoid probabilities and labels of every row of `loader`, as numpy arrays."""
    model.eval()
    probs, targets = [], []
    for x, y in tqdm(loader, desc="predict"):
        probs.append(torch.sigmoid(model(x.to(model.device))).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def f1_report(y_true, probs, threshold=0.5):
    """F1 of each class plus macro F1."""
    y_pred = (probs > threshold).astype(int)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return pd.Series(np.append(per_class, per_class.mean()), index=[*CLASSES, "macro"], name="f1")

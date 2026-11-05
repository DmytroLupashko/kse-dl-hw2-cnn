"""Decisions of submission: thresholds per class on OOF rows, test probs of fold models, mean of ensemble members.

`labels`, `probs`: arrays (rows, classes), `CLASSES` order. Files, naming, upload: `helpers.submission`, `helpers.ensemble`.
"""

import numpy as np

from core.augment import Validation
from core.dataset import MultilabelImageDataset
from core.metrics import best_f1, plateau_f1
from helpers.config import CLASSES, TEST_DIR


def single_thresholds(labels, probs, fixed=None):
    """One model: `best_f1` threshold per class on pooled OOF rows (`fixed` given: one value per class instead)."""

    if fixed is not None:
        return np.array(fixed, dtype=float)

    return best_f1(labels, probs, CLASSES)["threshold"].to_numpy()


def ensemble_thresholds(labels, probs, margin=0.05):
    """Ensemble: `plateau_f1` (`margin`) threshold per class on ensemble OOF probs."""

    return plateau_f1(labels, probs, CLASSES, margin)["threshold"].to_numpy()


def test_probs(models, test, data):
    """Mean sigmoid probs over fold `models` (generator: one in memory at a time), one row per row of `test`;
    `data`: `data` section of training config (`img_size`, `batch_size`, `num_workers`)."""

    loader = MultilabelImageDataset(
        test, TEST_DIR, Validation(data["img_size"]).compose(), is_test=True
    ).loader(data["batch_size"], data["num_workers"])
    fold_probs = [model.predict_probs(loader)[0] for model in models]

    return sum(fold_probs) / len(fold_probs)


def mean_probs(probs):
    """Plain mean of members' probs arrays, same rows."""

    return sum(probs) / len(probs)


def aligned_oofs(oofs):
    """Members' OOF frames, rows sorted; members must share rows and folds."""

    oofs = [o.sort_index() for o in oofs]

    if not all(
        o.index.equals(oofs[0].index) and o["fold"].equals(oofs[0]["fold"])
        for o in oofs
    ):
        raise ValueError("members' OOF rows differ")

    return oofs

"""Submission from trained CV checkpoints: test probs averaged over folds (ensembles: mean or logistic regression stacking over models), per-class thresholds tuned on OOF rows.

CLI: `python -m core.submission <config.toml> [--submit]`
Notebook: `make_submission(Config.from_toml(path))`.
CSV saved to `data/submissions/<experiment>__<name>.csv`, unthresholded test probs next to it as `.parquet` (for ensembling).
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import CLASSES, SUBMISSIONS, TEST_DIR
from .cv import PROB_COLS, Config, CVResult
from .data import MultilabelImageDataset, make_loader, read_test, val_tfms
from .predict import best_f1, plateau_f1, predict_probs

COMPETITION = "cats-and-dogs"
LOGIT_EPS = 1e-6  # probs clipped before logit: 0 and 1 give infinite logits (exp39)


def save_submission(probs, path, thresholds, test=None):
    """`image_id` plus one 0/1 column per class, rows in `test` order (default test.csv); `thresholds`: scalar or one per class."""
    submission = (read_test() if test is None else test)[["image_id"]].copy()
    if len(probs) != len(submission):
        raise ValueError(f"{len(probs)} predictions for {len(submission)} test images")

    submission[CLASSES] = (probs >= thresholds).astype(int)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"saved {path}: {len(submission):,} rows")
    return submission


def test_probs(config, test=None):
    """Mean sigmoid probs over fold models of `config.folds`, one row per row of `test` (default test.csv)."""
    test = read_test() if test is None else test
    ds = MultilabelImageDataset(test, TEST_DIR, val_tfms(config.img_size))
    loader = make_loader(ds, config.batch_size, config.num_workers)
    fold_probs = [predict_probs(config.fold_model(fold), loader)[0] for fold in config.folds]
    return sum(fold_probs) / len(fold_probs)


def make_submission(config, test=None, path=None):
    """Writes test probs and thresholded CSV of `config`'s saved CV run; returns CSV path.
    Thresholds: `best_f1` on pooled OOF rows of all trained folds. `test` subset and `path`: smoke runs."""
    result = CVResult.load(config)
    config = result.config  # stored config: CLI overrides used at training kept
    thresholds = best_f1(result.labels, result.probs)["threshold"].to_numpy()
    print(pd.Series(thresholds, index=CLASSES, name="threshold").round(3))

    test = read_test() if test is None else test
    probs = test_probs(config, test)
    path = Path(path or SUBMISSIONS / f"{config.experiment}__{config.name}.csv")
    save_submission(probs, path, thresholds, test)
    out = test[["image_id"]].copy()
    out[PROB_COLS] = probs
    out.to_parquet(path.with_suffix(".parquet"))
    return path


def member_oofs(configs):
    """OOF frames of `configs`' saved CV runs, rows sorted; members must share rows and folds."""
    oofs = [CVResult.load(c).oof.sort_index() for c in configs]
    if not all(o.index.equals(oofs[0].index) and o["fold"].equals(oofs[0]["fold"]) for o in oofs):
        raise ValueError("members' OOF rows differ")
    return oofs


def logit(probs):
    probs = np.clip(probs, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(probs / (1 - probs))


def stacked_probs(lr, features, normalize=False):
    """Probs of fitted `lr`; `normalize`: whole logit (weights and intercept) divided by sum of member weights,
    so members' weights sum to 1 and only their mix is kept, not the sharpening."""
    z = lr.decision_function(features)
    return 1 / (1 + np.exp(-z / lr.coef_[0].sum())) if normalize else 1 / (1 + np.exp(-z))


def cross_fold_stacking(features, labels, folds, C, names, normalize=False):
    """Per class L2 `LogisticRegression(C=C)` on `features[:, :, class]` (members' logits): each fold predicted by
    model fit on other folds (`normalize`: see `stacked_probs`). Returns stacked probs and raw weights
    (rows `(class, fold)`, columns `names` + intercept)."""
    stacked, rows = np.zeros(labels.shape), []
    for c in range(len(CLASSES)):
        for fold in np.unique(folds):
            rest, part = folds != fold, folds == fold
            lr = LogisticRegression(C=C).fit(features[rest, :, c], labels[rest, c])
            stacked[part, c] = stacked_probs(lr, features[part, :, c], normalize)
            rows.append({"class": CLASSES[c], "fold": fold, **dict(zip(names, lr.coef_[0])), "intercept": lr.intercept_[0]})
    return stacked, pd.DataFrame(rows).set_index(["class", "fold"])


def make_ensemble_submission(configs, path, test=None, margin=0.05):
    """Plain mean ensemble of `configs`' saved CV runs: OOF and test probs averaged over members, one threshold
    per class by `plateau_f1` on mean OOF probs; writes CSV `path` and test probs `.parquet` next to it."""
    oofs = member_oofs(configs)
    oof_probs = sum(o[PROB_COLS].to_numpy() for o in oofs) / len(oofs)
    thresholds = plateau_f1(oofs[0][CLASSES].to_numpy(), oof_probs, margin)["threshold"].to_numpy()
    print(pd.Series(thresholds, index=CLASSES, name="threshold").round(3))

    test = read_test() if test is None else test
    probs = sum(test_probs(c, test) for c in configs) / len(configs)
    path = Path(path)
    save_submission(probs, path, thresholds, test)
    out = test[["image_id"]].copy()
    out[PROB_COLS] = probs
    out.to_parquet(path.with_suffix(".parquet"))
    return path


def make_stacking_submission(configs, path, C, test=None, margin=0.05, normalize=False):
    """Logistic regression stacking of `configs`' saved CV runs: per class one L2 `LogisticRegression(C=C)` on members'
    logits of that class (`normalize`: see `stacked_probs`). Thresholds: `plateau_f1` on cross-fold stacked OOF probs (each fold by model fit on other
    folds). Test: model refit on all OOF rows, applied to members' test logits; writes CSV `path` and `.parquet`."""
    oofs = member_oofs(configs)
    labels, folds = oofs[0][CLASSES].to_numpy(), oofs[0]["fold"].to_numpy()
    features = np.stack([logit(o[PROB_COLS].to_numpy()) for o in oofs], axis=1)  # rows, members, classes
    stacked, _ = cross_fold_stacking(features, labels, folds, C, range(len(configs)), normalize)
    thresholds = plateau_f1(labels, stacked, margin)["threshold"].to_numpy()
    print(pd.Series(thresholds, index=CLASSES, name="threshold").round(3))

    test = read_test() if test is None else test
    test_features = np.stack([logit(test_probs(c, test)) for c in configs], axis=1)
    probs = np.column_stack([
        stacked_probs(LogisticRegression(C=C).fit(features[:, :, c], labels[:, c]), test_features[:, :, c], normalize)
        for c in range(len(CLASSES))
    ])
    path = Path(path)
    save_submission(probs, path, thresholds, test)
    out = test[["image_id"]].copy()
    out[PROB_COLS] = probs
    out.to_parquet(path.with_suffix(".parquet"))
    return path


def submit(path, message):
    """Uploads CSV `path` to Kaggle competition."""
    subprocess.run(["kaggle", "competitions", "submit", COMPETITION, "-f", str(path), "-m", message], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--submit", action="store_true", help="upload CSV to Kaggle")
    args = parser.parse_args()
    config = Config.from_toml(args.config)
    path = make_submission(config)
    if args.submit:
        submit(path, f"{config.experiment} / {config.name}, folds {config.folds}, OOF best-F1 thresholds")

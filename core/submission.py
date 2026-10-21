"""Submission from trained CV checkpoints: thresholds on OOF rows, test probs, thresholded CSV.

`Single(config)`: test probs averaged over folds, best-F1 thresholds. Ensembles of configs, plateau-F1 thresholds:
`MeanEnsemble` (mean of members), `Stacking` (per class logistic regression on members' logits).
CLI: `python -m core.submission <config.toml> [--submit]`
Notebook: `Single(Config.from_toml(path)).write()`.
CSV saved to `path` (`Single` default `data/submissions/<experiment>__<name>.csv`), unthresholded test probs next to it
as `.parquet` (for ensembling).
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .augment import Validation
from .config import CLASSES, SUBMISSIONS, TEST_DIR
from .cv import PROB_COLS, Config, CVResult
from .data import MultilabelImageDataset, read_test
from .scores import Scores


class Submission:
    """CSV `path` of thresholded test probs. Subclass: `thresholds()` per class (from OOF rows) and `test_probs(test)`."""

    COMPETITION = "cats-and-dogs"

    def __init__(self, path):
        self.path = Path(path)

    def thresholds(self):
        raise NotImplementedError

    def test_probs(self, test):
        raise NotImplementedError

    def write(self, test=None):
        """Thresholds, then probs of `test` rows (default test.csv; subset: smoke runs); writes CSV `path` (`image_id` plus
        one 0/1 column per class, rows in `test` order) and probs `.parquet`; returns `path`."""
        thresholds = self.thresholds()
        print(pd.Series(thresholds, index=CLASSES, name="threshold").round(3))
        test = read_test() if test is None else test
        probs = self.test_probs(test)
        submission = test[["image_id"]].copy()
        if len(probs) != len(submission):
            raise ValueError(f"{len(probs)} predictions for {len(submission)} test images")
        submission[CLASSES] = (probs >= thresholds).astype(int)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(self.path, index=False)
        print(f"saved {self.path}: {len(submission):,} rows")
        out = submission[["image_id"]].copy()
        out[PROB_COLS] = probs
        out.to_parquet(self.path.with_suffix(".parquet"))
        return self.path

    def submit(self, message):
        """Uploads written CSV to Kaggle competition."""
        subprocess.run(["kaggle", "competitions", "submit", self.COMPETITION, "-f", str(self.path), "-m", message], check=True)


class Single(Submission):
    """One model: saved CV run of `config`, test probs averaged over fold models, `best_f1` thresholds on pooled OOF rows.
    Stored config used: CLI overrides at training kept."""

    def __init__(self, config, path=None):
        self.result = CVResult.load(config)
        self.config = self.result.config
        super().__init__(path or SUBMISSIONS / f"{self.config.experiment}__{self.config.name}.csv")

    def thresholds(self):
        return self.result.scores.best_f1()["threshold"].to_numpy()

    def test_probs(self, test):
        """Mean sigmoid probs over fold models of `config.folds`, one row per row of `test`."""
        config = self.config
        loader = MultilabelImageDataset(test, TEST_DIR, Validation(config.img_size).compose()).loader(
            config.batch_size, config.num_workers
        )
        fold_probs = [config.fold_model(fold).predict_probs(loader)[0] for fold in config.folds]
        return sum(fold_probs) / len(fold_probs)


class Ensemble(Submission):
    """Members: `Single` of each of `configs`, same OOF rows and folds. Thresholds: `plateau_f1` (`margin`) on
    ensemble OOF probs (`oof_probs`)."""

    def __init__(self, configs, path, margin=0.05):
        super().__init__(path)
        self.members = [Single(c) for c in configs]
        self.margin = margin
        self.oofs = self.member_oofs(configs)
        self.labels = self.oofs[0][CLASSES].to_numpy()
        self.folds = self.oofs[0]["fold"].to_numpy()

    @staticmethod
    def member_oofs(configs):
        """OOF frames of `configs`' saved CV runs, rows sorted; members must share rows and folds."""
        oofs = [CVResult.load(c).oof.sort_index() for c in configs]
        if not all(o.index.equals(oofs[0].index) and o["fold"].equals(oofs[0]["fold"]) for o in oofs):
            raise ValueError("members' OOF rows differ")
        return oofs

    def oof_probs(self):
        raise NotImplementedError

    def thresholds(self):
        return Scores(self.labels, self.oof_probs()).plateau_f1(self.margin)["threshold"].to_numpy()


class MeanEnsemble(Ensemble):
    """Plain mean of members' OOF and test probs."""

    def oof_probs(self):
        return sum(o[PROB_COLS].to_numpy() for o in self.oofs) / len(self.oofs)

    def test_probs(self, test):
        return sum(m.test_probs(test) for m in self.members) / len(self.members)


class Stacking(Ensemble):
    """Per class one L2 `LogisticRegression(C=C)` on members' logits of that class (`normalize`: see `stacked_probs`).
    OOF probs: each fold by model fit on other folds. Test: model refit on all OOF rows."""

    LOGIT_EPS = 1e-6  # probs clipped before logit: 0 and 1 give infinite logits (exp39)

    def __init__(self, configs, path, C, margin=0.05, normalize=False):
        super().__init__(configs, path, margin)
        self.C = C
        self.normalize = normalize
        self.features = np.stack([self.logit(o[PROB_COLS].to_numpy()) for o in self.oofs], axis=1)  # rows, members, classes

    @classmethod
    def logit(cls, probs):
        probs = np.clip(probs, cls.LOGIT_EPS, 1 - cls.LOGIT_EPS)
        return np.log(probs / (1 - probs))

    @staticmethod
    def stacked_probs(lr, features, normalize=False):
        """Probs of fitted `lr`; `normalize`: whole logit (weights and intercept) divided by sum of member weights,
        so members' weights sum to 1 and only their mix is kept, not the sharpening."""
        z = lr.decision_function(features)
        return 1 / (1 + np.exp(-z / lr.coef_[0].sum())) if normalize else 1 / (1 + np.exp(-z))

    @classmethod
    def cross_fold(cls, features, labels, folds, C, names, normalize=False):
        """Per class L2 `LogisticRegression(C=C)` on `features[:, :, class]` (members' logits): each fold predicted by
        model fit on other folds. Returns stacked probs and raw weights (rows `(class, fold)`, columns `names` + intercept)."""
        stacked, rows = np.zeros(labels.shape), []
        for c in range(len(CLASSES)):
            for fold in np.unique(folds):
                rest, part = folds != fold, folds == fold
                lr = LogisticRegression(C=C).fit(features[rest, :, c], labels[rest, c])
                stacked[part, c] = cls.stacked_probs(lr, features[part, :, c], normalize)
                rows.append({"class": CLASSES[c], "fold": fold, **dict(zip(names, lr.coef_[0])), "intercept": lr.intercept_[0]})
        return stacked, pd.DataFrame(rows).set_index(["class", "fold"])

    def oof_probs(self):
        stacked, _ = self.cross_fold(self.features, self.labels, self.folds, self.C, range(len(self.members)), self.normalize)
        return stacked

    def test_probs(self, test):
        test_features = np.stack([self.logit(m.test_probs(test)) for m in self.members], axis=1)
        return np.column_stack([
            self.stacked_probs(
                LogisticRegression(C=self.C).fit(self.features[:, :, c], self.labels[:, c]), test_features[:, :, c], self.normalize
            )
            for c in range(len(CLASSES))
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--submit", action="store_true", help="upload CSV to Kaggle")
    args = parser.parse_args()
    config = Config.from_toml(args.config)
    submission = Single(config)
    submission.write()
    if args.submit:
        submission.submit(f"{config.experiment} / {config.name}, folds {config.folds}, OOF best-F1 thresholds")

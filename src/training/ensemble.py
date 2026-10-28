"""Ensembles of saved CV runs, plateau-F1 thresholds on ensemble OOF probs; written as submissions (`training.submission`).

`MeanEnsemble` (mean of members), `Stacking` (per class logistic regression on members' logits).
Notebook: `MeanEnsemble([config path, ...], path).write()`.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from analysis.thresholds import plateau_f1
from .config import CLASSES
from .cv import CVResult
from .runs import PROB_COLS
from .submission import Single, Submission


class Ensemble(Submission):
    """Members: `Single` of each model config path of `configs`, same OOF rows and folds. Thresholds: `plateau_f1` (`margin`) on
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

        if not all(
            o.index.equals(oofs[0].index) and o["fold"].equals(oofs[0]["fold"])
            for o in oofs
        ):
            raise ValueError("members' OOF rows differ")

        return oofs

    def thresholds(self):
        return plateau_f1(self.labels, self.oof_probs(), CLASSES, self.margin)[
            "threshold"
        ].to_numpy()


class MeanEnsemble(Ensemble):
    """Plain mean of members' OOF and test probs."""

    def oof_probs(self):
        return sum(o[PROB_COLS].to_numpy() for o in self.oofs) / len(self.oofs)

    def test_probs(self, test):
        return sum(m.test_probs(test) for m in self.members) / len(self.members)


class Stacking(Ensemble):
    """Per class one L2 `LogisticRegression(C=C)` on members' logits of that class (`normalize`: see `stacked_probs`).
    OOF probs: each fold by model fit on other folds. Test: model refit on all OOF rows."""

    def __init__(self, configs, path, C, margin=0.05, normalize=False):
        super().__init__(configs, path, margin)
        self.C = C
        self.normalize = normalize
        self.features = np.stack(
            [self.logit(o[PROB_COLS].to_numpy()) for o in self.oofs], axis=1
        )  # rows, members, classes

    @staticmethod
    def logit(probs):
        probs = np.clip(probs, 1e-6, 1 - 1e-6)  # 0 and 1 give infinite logits (exp39)

        return np.log(probs / (1 - probs))

    @staticmethod
    def stacked_probs(lr, features, normalize):
        """Probs of fitted `lr`; `normalize`: whole logit (weights and intercept) divided by sum of member weights,
        so members' weights sum to 1 and only their mix is kept, not the sharpening."""
        z = lr.decision_function(features)

        return (
            1 / (1 + np.exp(-z / lr.coef_[0].sum()))
            if normalize
            else 1 / (1 + np.exp(-z))
        )

    @classmethod
    def cross_fold(cls, features, labels, folds, C, names, normalize=False):
        """Per class L2 `LogisticRegression(C=C)` on `features[:, :, class]` (members' logits): each fold predicted by
        model fit on other folds. Returns stacked probs and raw weights (rows `(class, fold)`, columns `names` + intercept)."""
        stacked, rows = np.zeros(labels.shape), []

        for c in range(len(CLASSES)):
            for fold in np.unique(folds):
                rest, part = folds != fold, folds == fold
                lr = LogisticRegression(C=C).fit(features[rest, :, c], labels[rest, c])
                stacked[part, c] = cls.stacked_probs(
                    lr, features[part, :, c], normalize
                )
                rows.append(
                    {
                        "class": CLASSES[c],
                        "fold": fold,
                        **dict(zip(names, lr.coef_[0])),
                        "intercept": lr.intercept_[0],
                    }
                )

        return stacked, pd.DataFrame(rows).set_index(["class", "fold"])

    def oof_probs(self):
        stacked, _ = self.cross_fold(
            self.features,
            self.labels,
            self.folds,
            self.C,
            range(len(self.members)),
            self.normalize,
        )

        return stacked

    def test_probs(self, test):
        test_features = np.stack(
            [self.logit(m.test_probs(test)) for m in self.members], axis=1
        )

        return np.column_stack(
            [
                self.stacked_probs(
                    LogisticRegression(C=self.C).fit(
                        self.features[:, :, c], self.labels[:, c]
                    ),
                    test_features[:, :, c],
                    self.normalize,
                )
                for c in range(len(CLASSES))
            ]
        )

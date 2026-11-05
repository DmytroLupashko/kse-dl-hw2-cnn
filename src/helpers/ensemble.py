"""Ensembles of saved CV runs, plateau-F1 thresholds on ensemble OOF probs; written as submissions (`helpers.submission`).
Decisions: `core.predict`.

`MeanEnsemble`: mean of members.
Notebook: `MeanEnsemble([config path, ...], path).write()`.
"""

from core import predict
from .config import CLASSES
from .cv import CVResult
from .tracking import PROB_COLS
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

        return predict.aligned_oofs([CVResult.load(c).oof for c in configs])

    def thresholds(self):
        return predict.ensemble_thresholds(self.labels, self.oof_probs(), self.margin)


class MeanEnsemble(Ensemble):
    """Plain mean of members' OOF and test probs."""

    def oof_probs(self):
        return predict.mean_probs([o[PROB_COLS].to_numpy() for o in self.oofs])

    def test_probs(self, test):
        return predict.mean_probs([m.test_probs(test) for m in self.members])

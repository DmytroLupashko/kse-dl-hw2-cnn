"""Ensembles of saved CV runs, plateau-F1 thresholds on ensemble OOF probs; written as submissions (`training.submission`).

`MeanEnsemble`: mean of members.
Notebook: `MeanEnsemble([config path, ...], path).write()`.
"""

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

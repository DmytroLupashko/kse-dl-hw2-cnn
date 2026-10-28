"""`metrics`, `thresholds` on hand-computed cases. Run: `python -m unittest discover -s tests`."""

import unittest

import numpy as np

from core.analysis.metrics import ap, f1_at, pr_curve
from core.analysis.thresholds import best_f1, cross_fold_f1, plateau_f1

CLASSES = ["A", "B"]
# A: F1 per threshold (0.1, 0.4, 0.8, 0.9) = (2/3, 0.8, 0.5, 2/3); B: (0.2, 0.3, 0.6, 0.7) = (0.4, 0.5, 2/3, 1)
Y = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
PROBS = np.array([[0.9, 0.2], [0.8, 0.7], [0.4, 0.6], [0.1, 0.3]])


class TestMetrics(unittest.TestCase):
    def test_ap(self):
        np.testing.assert_allclose(ap(Y, PROBS, CLASSES), [0.5 + 0.5 * 2 / 3, 1.0])

    def test_pr_curve(self):
        precision, recall, thr, f1 = pr_curve(Y[:, 0], PROBS[:, 0])
        np.testing.assert_allclose(thr, [0.1, 0.4, 0.8, 0.9])
        np.testing.assert_allclose(f1, [2 / 3, 0.8, 0.5, 2 / 3])

    def test_f1_at(self):
        table = f1_at(Y, PROBS, [0.5, 0.5], CLASSES)
        np.testing.assert_allclose(table[["precision", "recall", "f1"]], [[0.5, 0.5, 0.5], [0.5, 1, 2 / 3]])


class TestThresholds(unittest.TestCase):
    def test_best_f1(self):
        table = best_f1(Y, PROBS, CLASSES)
        np.testing.assert_allclose(table, [[0.4, 2 / 3, 1, 0.8], [0.7, 1, 1, 1]])

    def test_plateau_f1(self):
        # margin 0.35: A plateau 0.1..0.9 (F1 >= 0.45), B 0.6..0.7 (F1 >= 0.65)
        table = plateau_f1(Y, PROBS, CLASSES, margin=0.35)
        np.testing.assert_allclose(table, [[0.5, 0.1, 0.9, 0.5], [0.65, 0.6, 0.7, 1]])

    def test_cross_fold_f1(self):
        # two equal classes (one column reads as binary in `f1_score`); fold 0 rows tuned on fold 1 and back
        y = np.repeat([[1], [0], [1], [0], [1], [0], [0], [1]], 2, axis=1)
        probs = np.repeat([[0.9], [0.8], [0.4], [0.1], [0.7], [0.6], [0.2], [0.3]], 2, axis=1)
        folds = [0, 0, 0, 0, 1, 1, 1, 1]
        # best: thresholds 0.3 (fold 0), 0.4 (fold 1); 3 TP of 5 predicted, 4 positives
        np.testing.assert_allclose(cross_fold_f1(y, probs, folds, CLASSES), [6 / 9] * 2)
        # margin 0.05: plateau is peak only, same as best
        np.testing.assert_allclose(cross_fold_f1(y, probs, folds, CLASSES, margin=0.05), [6 / 9] * 2)
        # margin 0.35: thresholds 0.45 (fold 0), 0.5 (fold 1); 2 TP of 4 predicted
        np.testing.assert_allclose(cross_fold_f1(y, probs, folds, CLASSES, margin=0.35), [0.5] * 2)


if __name__ == "__main__":
    unittest.main()

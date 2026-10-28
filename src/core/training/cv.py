"""Cross-validation results of one model config: OOF probs, per-fold AP, MLflow histories, fold checkpoints.

Training: one LightningCLI run per fold (`core.training.cli`), outputs in run folders (`core.training.runs`).
Notebook: `CVResult.load(path)` reads finished folds of model config `path`.
"""

import dataclasses
from pathlib import Path

import pandas as pd

from core.analysis.metrics import ap
from core.analysis.thresholds import best_f1
from core.training.config import CLASSES, DEVICE
from core.training.model import LitCNNMultiLabel
from core.training.runs import PROB_COLS, load_runs
from core.training.tracking import curves

HISTORY = {  # MLflow epoch metric: (metric, split) of `CVResult.history`
    "train_loss_epoch": ("loss", "train"),
    "val_loss": ("loss", "validation"),
    "train_map": ("mAP", "train"),
    "val_map": ("mAP", "validation"),
}


@dataclasses.dataclass
class CVResult:
    config: dict  # resolved training config: `experiment`, `name`, `data` section (`img_size`, `batch_size`, ...)
    oof: pd.DataFrame  # validation rows of `folds`: fold, labels, `p_<class>` probs
    run_ids: dict[int, str]  # fold: MLflow run id
    run_dirs: dict[int, Path]  # fold: folder of its checkpoint

    @property
    def folds(self):
        return sorted(self.run_ids)

    @property
    def fold_ap(self):
        """AP per class plus mAP, one row per fold."""
        rows = {}

        for fold, y, probs in self._per_fold():
            aps = ap(y, probs, CLASSES)
            rows[fold] = {**aps, "mAP": aps.mean()}

        return pd.DataFrame(rows).T.rename_axis("fold")

    @property
    def labels(self):
        return self.oof[CLASSES].to_numpy()

    @property
    def probs(self):
        return self.oof[PROB_COLS].to_numpy()

    def mean_curve(self, key):
        """Per-epoch metric `key` of fold runs, averaged over folds."""
        return curves(self.run_ids, [key]).groupby("epoch")["value"].mean().to_numpy()

    def history(self):
        """Per-epoch `HISTORY` metrics of every fold run, long format (fold, epoch, metric, split, value)."""
        names = pd.DataFrame(
            HISTORY.values(), index=list(HISTORY), columns=["metric", "split"]
        )

        return curves(self.run_ids, HISTORY).join(names, on="key")[
            ["fold", "epoch", "metric", "split", "value"]
        ]

    def fold_scores(self):
        """AP per class, mAP and Macro F1 (best-F1 thresholds tuned on same fold rows), one row per fold."""
        macro_f1 = {
            fold: best_f1(y, probs, CLASSES)["f1"].mean()
            for fold, y, probs in self._per_fold()
        }

        return self.fold_ap.assign(**{"Macro F1": pd.Series(macro_f1)})

    def _per_fold(self):
        """(fold, labels, probs) of its OOF rows, one per fold."""
        for fold, part in self.oof.groupby("fold"):
            yield fold, part[CLASSES].to_numpy(), part[PROB_COLS].to_numpy()

    def only(self, folds):
        """Copy restricted to `folds`."""
        return dataclasses.replace(
            self,
            oof=self.oof[self.oof["fold"].isin(folds)],
            run_ids={fold: self.run_ids[fold] for fold in folds},
            run_dirs={fold: self.run_dirs[fold] for fold in folds},
        )

    def fold_model(self, fold):
        """Trained `LitCNNMultiLabel` of `fold` from its checkpoint, on `DEVICE`."""
        return LitCNNMultiLabel.load(
            next(self.run_dirs[fold].glob("*.ckpt")), map_location=DEVICE
        )

    @classmethod
    def load(cls, path):
        """Finished folds of model config `path` (its `experiment`, `name` locate them)."""
        return cls(*load_runs(path))

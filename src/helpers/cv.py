"""Cross-validation results of one model config: OOF probs, per-fold AP, MLflow histories, fold checkpoints.

Training: one LightningCLI run per fold (`core.cli`), outputs in run folders (`helpers.tracking`).
Notebook: `CVResult.load(path)` reads finished folds of model config `path`.
"""

import dataclasses
import importlib
from pathlib import Path

import pandas as pd
import torch

from core.metrics import ap, best_f1
from helpers.config import CLASSES, DEVICE
from core.model import LitCNNMultiLabel
from helpers.tracking import PROB_COLS, curves, load_runs

def load_model(ckpt, map_location=None):
    """Trained `LitCNNMultiLabel` of checkpoint `ckpt`. Checkpoint before LightningCLI: network from its `model` block
    (`_target_` plus kwargs), weights from state dict."""

    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)

    if "model" not in checkpoint["hyper_parameters"]:
        return LitCNNMultiLabel.load_from_checkpoint(ckpt, map_location=map_location)

    block = dict(checkpoint["hyper_parameters"]["model"])
    module, _, name = block.pop("_target_").rpartition(".")

    if module.startswith(("core.", "training.")):  # project nets moved here since
        module = "core.model"

    net = getattr(importlib.import_module(module), name)(**block, num_classes=len(CLASSES))
    model = LitCNNMultiLabel(net, pos_weight=[1.0] * len(CLASSES))  # placeholder: state dict overwrites loss buffer
    model.load_state_dict(checkpoint["state_dict"])

    return model.to(map_location) if map_location else model


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

        aps = pd.DataFrame({fold: ap(y, probs, CLASSES) for fold, y, probs in self._per_fold()}).T

        return aps.assign(mAP=aps.mean(axis=1)).rename_axis("fold")

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

        names = pd.DataFrame(HISTORY.values(), index=list(HISTORY), columns=["metric", "split"])

        return curves(self.run_ids, HISTORY).join(names, on="key")[["fold", "epoch", "metric", "split", "value"]]

    def fold_scores(self):
        """AP per class, mAP and Macro F1 (best-F1 thresholds tuned on same fold rows), one row per fold."""

        macro_f1 = {fold: best_f1(y, probs, CLASSES)["f1"].mean() for fold, y, probs in self._per_fold()}

        return self.fold_ap.assign(**{"Macro F1": pd.Series(macro_f1)})

    def _per_fold(self):
        """(fold, labels, probs) of its OOF rows, one per fold."""

        for fold, part in self.oof.groupby("fold"):
            yield fold, part[CLASSES].to_numpy(), part[PROB_COLS].to_numpy()

    def fold_model(self, fold):
        """Trained `LitCNNMultiLabel` of `fold` from its checkpoint, on `DEVICE`."""

        return load_model(next(self.run_dirs[fold].glob("*.ckpt")), map_location=DEVICE)

    @classmethod
    def load(cls, path):
        """Finished folds of model config `path` (its `experiment`, `name` locate them)."""

        return cls(*load_runs(path))

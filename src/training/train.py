"""Trainer callbacks of `training.cli`: mAP (`MeanAP`), run folder outputs (`RunCheckpoint`), progress bars."""

import sys
from pathlib import Path

from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from torchmetrics.classification import MultilabelAveragePrecision

from analysis.metrics import ap
from .config import CLASSES
from .model import LitCNNMultiLabel
from .runs import PROB_COLS, save_run


class MeanAP(Callback):
    """Macro AP per epoch of `probs`, `labels` returned by model steps: `train_map`, `val_map` (`RunCheckpoint` monitor)."""

    def on_fit_start(self, trainer, pl_module):
        self.metrics = {
            split: MultilabelAveragePrecision(
                num_labels=len(CLASSES), average="macro"
            ).to(pl_module.device)
            for split in ("train", "val")
        }

    def _update(self, split, outputs):
        self.metrics[split].update(outputs["probs"], outputs["labels"].int())

    def _log(self, split, pl_module):
        pl_module.log(f"{split}_map", self.metrics[split].compute(), prog_bar=True)
        self.metrics[split].reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._update("train", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log("train", pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._update("val", outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log("val", pl_module)


class RunCheckpoint(ModelCheckpoint):
    """Run folder `trainer.default_root_dir` (`training.runs.run_dir`): best checkpoint by `monitor` (min when name
    has "loss", else max; no validation fold: latest epoch), then `oof.parquet` (validation rows, probs of best
    checkpoint; best AP per class logged to MLflow) and `run.json` (MLflow run id)."""

    def __init__(self, monitor="val_map"):
        mode = "min" if "loss" in monitor else "max"
        # weights only: loaded for prediction only, optimizer state triples size
        super().__init__(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_weights_only=True,
            filename="{epoch:02d}",
        )

    def setup(self, trainer, pl_module, stage):
        self.dirpath = trainer.default_root_dir

        if trainer.datamodule.fold is None:
            self.monitor = None

        super().setup(trainer, pl_module, stage)

    def on_fit_end(self, trainer, pl_module):
        dm, val = trainer.datamodule, None

        if dm.fold is not None:
            best = LitCNNMultiLabel.load(
                self.best_model_path, map_location=pl_module.device
            )
            probs, y = best.predict_probs(dm.val_dataloader())
            trainer.logger.log_metrics(
                {f"best_ap_{c}": v for c, v in ap(y, probs, CLASSES).items()}
            )
            val = dm.df.loc[dm.df["fold"] == dm.fold, ["fold", *CLASSES]].copy()
            val[PROB_COLS] = probs

        save_run(Path(self.dirpath), trainer.logger.run_id, val)


class EpochBars(TQDMProgressBar):
    """One tqdm bar per epoch, kept on screen; no validation or sanity check bar.
    Bar closed at next epoch start: `train_map` logged in epoch-end hook of `MeanAP`, maybe after this one. Then line per checkpoint save."""

    def __init__(self):
        super().__init__(leave=True, refresh_rate=1)
        self._open = False
        self._best = ""

    def init_validation_tqdm(self):
        return Tqdm(disable=True)

    def init_sanity_tqdm(self):
        return Tqdm(disable=True)

    def on_train_start(self, *_):
        pass  # bar created per epoch

    def on_train_epoch_start(self, trainer, pl_module):
        self._finish_epoch(trainer, pl_module)
        # own bar: base creates it without total first, which prints empty "Training 0/?" bar
        self.train_progress_bar = Tqdm(
            desc=f"Epoch {trainer.current_epoch}",
            total=self.total_train_batches,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )
        self._open = True

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)

        return metrics

    def on_train_epoch_end(self, *_):
        pass

    def on_train_end(self, trainer, pl_module):
        self._finish_epoch(trainer, pl_module)

    def _finish_epoch(self, trainer, pl_module):
        if not self._open:
            return

        self._open = False
        self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        self.train_progress_bar.close()
        best = trainer.checkpoint_callback.best_model_path

        if best != self._best:
            self._best = best
            print(f"    checkpoint saved: {Path(best).name}", flush=True)

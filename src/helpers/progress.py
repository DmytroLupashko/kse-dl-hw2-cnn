"""Training progress display of `core.cli`: one tqdm bar per epoch (`EpochBars`)."""

import sys
from pathlib import Path

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm


class EpochBars(TQDMProgressBar):
    """One tqdm bar per epoch, kept on screen; no validation or sanity check bar.
    Bar closed at next epoch start: epoch value of `train_map` computed after callback epoch-end hooks, this one included. Then line per checkpoint save."""

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

"""One training run, logged to MLflow, best checkpoint by `val_map`; finished runs reused."""
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.loggers import MLFlowLogger

from .config import CHECKPOINTS, MLFLOW_URI, SEED


def recipe_hash(lit_model, datamodule, epochs, trainer_kw):
    """Short hash of everything that sets trained weights.
    Not covered: code changes not visible in these values, e.g. loss in `LitCNNMultiLabel`."""
    dm = datamodule
    pos_weight = lit_model.pos_weight
    recipe = {
        "model": [type(lit_model).__qualname__, type(lit_model.model).__qualname__],
        "hparams": dict(lit_model.hparams),
        "pos_weight": None if pos_weight is None else pos_weight.tolist(),
        "data": int(pd.util.hash_pandas_object(dm.df, index=False).sum()),
        "fold": dm.fold,
        "batch_size": dm.batch_size,
        "tfms": [dm.train_tfms.to_dict(), dm.val_tfms.to_dict()],
        "epochs": epochs,
        "trainer_kw": trainer_kw,
        "seed": SEED,
    }
    if dm.oversample != 1:  # key absent for plain shuffling: hashes of earlier runs unchanged
        recipe["oversample"] = dm.oversample
    if dm.downsample != 1:
        recipe["downsample"] = dm.downsample
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[:12]


class EpochBars(TQDMProgressBar):
    """One tqdm bar per epoch, kept on screen; no validation or sanity check bar.
    Bar closed at next epoch start: `train_map` logged after epoch-end callbacks. Then line per checkpoint save."""

    def __init__(self):
        super().__init__(leave=True)
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


def train(lit_model, datamodule, *, experiment, run_name, epochs, reuse=True, **trainer_kw):
    """Fits `lit_model` as run `run_name` of MLflow experiment `experiment` (`cnn-<section>-<slug>`).
    Returns (best checkpoint path, MLflow run id).
    `reuse`: finished earlier run with same `experiment`, `run_name` and `recipe_hash` returned without
    training; changed recipe retrains and replaces it."""
    ckpt_dir = CHECKPOINTS / experiment / run_name
    done = ckpt_dir / "run.json"  # written only after fit finishes, so interrupted runs retrain
    recipe = recipe_hash(lit_model, datamodule, epochs, trainer_kw)
    if reuse and done.exists():
        run = json.loads(done.read_text())
        if run["recipe"] == recipe:
            best = next(ckpt_dir.glob("*.ckpt"))
            print(f"\n=== {run_name} | {experiment} | reused finished run ===\n    checkpoint: {best.name}")
            return str(best), run["run_id"]
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # interrupted or replaced run
    print(f"\n=== {run_name} | {experiment} | {epochs} epochs ===", flush=True)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)  # no device, tip, stop info lines
    pl.seed_everything(SEED, verbose=False)
    logger = MLFlowLogger(experiment_name=experiment, run_name=run_name, tracking_uri=MLFLOW_URI)
    logger.log_hyperparams({"recipe": recipe})
    has_val = datamodule.fold is not None
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_map" if has_val else None,
        mode="max",
        save_top_k=1,
        save_weights_only=True,  # only loaded for prediction; optimizer state triples size
        filename="{epoch:02d}-{val_map:.4f}" if has_val else "{epoch:02d}",
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step"), EpochBars()],
        enable_model_summary=False,
        accelerator="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        **trainer_kw,
    )
    trainer.fit(lit_model, datamodule=datamodule)
    done.write_text(json.dumps({"run_id": logger.run_id, "recipe": recipe}))
    return checkpoint_cb.best_model_path, logger.run_id

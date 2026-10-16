"""One training run, logged to MLflow, best checkpoint by `val_f1`."""
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from .config import CHECKPOINTS, MLFLOW_URI, SEED


def train(lit_model, datamodule, *, experiment, run_name, epochs, **trainer_kw):
    """Fits `lit_model` as run `run_name` of MLflow experiment `experiment` (`cnn-<section>-<slug>`).
    Returns the trainer; best checkpoint: `trainer.checkpoint_callback.best_model_path`."""
    pl.seed_everything(SEED, verbose=False)
    logger = MLFlowLogger(experiment_name=experiment, run_name=run_name, tracking_uri=MLFLOW_URI)
    has_val = datamodule.fold is not None
    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINTS / experiment / run_name,
        monitor="val_f1" if has_val else None,
        mode="max",
        save_top_k=1,
        filename="{epoch:02d}-{val_f1:.4f}" if has_val else "{epoch:02d}",
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
        accelerator="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        **trainer_kw,
    )
    trainer.fit(lit_model, datamodule=datamodule)
    return trainer

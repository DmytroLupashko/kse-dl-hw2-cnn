"""Lightning wrapper of any CNN with one logit per class: Adam, cosine schedule per batch, BCE, mAP. Trainer
callback of `core.cli`: run folder outputs (`RunCheckpoint`); progress bars: `helpers.progress`.

Networks: YAML `model.net` block (`class_path`, `init_args`) names `nn.Module` class or function returning one:
`timm_net`.
"""

import logging
import warnings
from pathlib import Path

import lightning.pytorch as pl
import timm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torchmetrics.classification import MultilabelAveragePrecision

from core.metrics import ap
from helpers.config import CLASSES
from helpers.tracking import PROB_COLS, save_run

# kept on purpose: checkpoint rebuilds `net` from its YAML block
warnings.filterwarnings("ignore", message="Attribute 'net' is an instance")


def timm_net(model_name: str, pretrained: bool = True) -> nn.Module:
    """timm model `model_name` with one logit per class; typed wrapper: `timm.create_model` takes `**kwargs`, which
    LightningCLI does not accept."""

    return timm.create_model(model_name, pretrained=pretrained, num_classes=len(CLASSES))


class LitCNNMultiLabel(pl.LightningModule):
    """`net`: CNN with one logit per class. Adam (`lr`), `CosineAnnealingLR` to `eta_min` stepped per batch, BCE with
    `pos_weight` to `pos_weight_power` (`core.cli` links `KaggleDataModule.pos_weight`; saved in checkpoint).
    Macro AP per epoch: `train_map`, `val_map` (`RunCheckpoint` monitor). Trained one: `helpers.cv.load_model`."""

    def __init__(
        self,
        net: nn.Module,
        pos_weight: list[float],
        lr: float = 1e-3,
        eta_min: float = 1e-5,
        pos_weight_power: float = 1.0,
    ):
        super().__init__()

        self.save_hyperparameters()  # under LightningCLI: YAML block of `net`, so checkpoint rebuilds it
        self.model = net
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32) ** pos_weight_power)

        self.train_map = MultilabelAveragePrecision(num_labels=len(CLASSES), average="macro")
        self.val_map = MultilabelAveragePrecision(num_labels=len(CLASSES), average="macro")

    def predict_probs(self, loader):
        """Sigmoid probabilities and labels of every row of `loader`, as numpy arrays (`trainer.predict`)."""

        logger = logging.getLogger("lightning.pytorch")
        level = logger.level
        logger.setLevel(logging.WARNING)  # no device info lines per call

        try:
            trainer = pl.Trainer(devices=1, logger=False, enable_checkpointing=False, enable_model_summary=False)
            batches = trainer.predict(self, loader)
        finally:
            logger.setLevel(level)

        probs, targets = zip(*batches)

        return torch.cat(probs).numpy(), torch.cat(targets).numpy()

    def predict_step(self, batch, batch_idx):
        x, y = batch

        return torch.sigmoid(self(x).float()).cpu(), y.cpu()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, split):
        """Loss logged as `<split>_loss`: per step and epoch for train, per epoch for val. Macro AP as `<split>_map`
        per epoch: Lightning computes, resets at epoch end."""

        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log(f"{split}_loss", loss, on_step=split == "train", on_epoch=True, prog_bar=True)

        metric = getattr(self, f"{split}_map")
        metric.update(torch.sigmoid(logits.float()).detach(), y.int())
        self.log(f"{split}_map", metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=self.hparams.eta_min,
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]


class RunCheckpoint(ModelCheckpoint):
    """Run folder `trainer.default_root_dir` (`helpers.tracking.run_dir`): best checkpoint by `monitor` (min when name
    has "loss", else max; no validation fold: latest epoch), then `oof.parquet` (validation rows, probs of best
    checkpoint; best AP per class logged to MLflow) and `run.json` (MLflow run id)."""

    def __init__(self, monitor="val_map"):
        # weights only: loaded for prediction only, optimizer state triples size
        super().__init__(
            monitor=monitor,
            mode="min" if "loss" in monitor else "max",
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
            best = LitCNNMultiLabel.load_from_checkpoint(self.best_model_path, map_location=pl_module.device)
            probs, y = best.predict_probs(dm.val_dataloader())
            trainer.logger.log_metrics({f"best_ap_{c}": v for c, v in ap(y, probs, CLASSES).items()})

            val = dm.df.loc[dm.df["fold"] == dm.fold, ["fold", *CLASSES]].copy()
            val[PROB_COLS] = probs

        save_run(Path(self.dirpath), trainer.logger.run_id, val)


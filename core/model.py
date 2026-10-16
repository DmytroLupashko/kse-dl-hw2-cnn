"""Lightning wrapper of any CNN with `num_classes` logits: BCE loss, macro F1, Adam + cosine schedule."""
import lightning.pytorch as pl
import timm
import torch
import torch.nn as nn
import torchmetrics

from .config import CLASSES


def macro_f1():
    return torchmetrics.F1Score(task="multilabel", num_labels=len(CLASSES), average="macro")


class LitCNNMultiLabel(pl.LightningModule):
    def __init__(self, model_class=timm.create_model, model_kwargs=None, initial_lr=1e-3, final_lr=1e-5,
                 threshold=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**(model_kwargs or {}))
        self.criterion = nn.BCEWithLogitsLoss()  # one independent sigmoid per class
        self.train_f1 = macro_f1()
        self.val_f1 = macro_f1()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, f1):
        x, y = batch
        logits = self(x)
        f1.update(torch.sigmoid(logits) > self.hparams.threshold, y.int())
        return self.criterion(logits, y)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_f1)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, self.val_f1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=self.hparams.final_lr
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

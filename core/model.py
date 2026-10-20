"""Lightning wrapper of any CNN with `num_classes` logits: weighted BCE loss, mAP, Adam + cosine schedule."""
import lightning.pytorch as pl
import numpy as np
import timm
import torch
import torch.nn as nn
import torchmetrics
import torchvision

from .config import CLASSES


def pos_weight(labels, power=0.75, row_weights=None):
    """BCE `pos_weight` per class: ((N - n_c) / n_c) ** power, from 0/1 `labels` of shape (N, classes).
    `row_weights`: draws per epoch of each row; N and n_c counted over draws."""
    labels = np.asarray(labels)
    row_weights = np.ones(len(labels)) if row_weights is None else np.asarray(row_weights)
    n_pos = row_weights @ labels
    return torch.tensor(((row_weights.sum() - n_pos) / n_pos) ** power, dtype=torch.float32)


def mean_ap():
    return torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(CLASSES), average="macro")


def alexnet_imagenet(num_classes):
    """torchvision AlexNet with ImageNet-1k weights (no timm version); last layer replaced for `num_classes`."""
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


TORCHVISION_MODELS = {"alexnet": alexnet_imagenet}  # name: builder taking `num_classes`, ImageNet weights


class LitCNNMultiLabel(pl.LightningModule):
    """`pos_weight`: tensor from `pos_weight()` on train part of fold; None means plain BCE.
    `load_from_checkpoint` needs same `model_class` and `pos_weight` passed again."""

    def __init__(self, model_class=timm.create_model, model_kwargs=None, initial_lr=1e-3, final_lr=1e-5,
                 pos_weight=None):
        super().__init__()
        self.save_hyperparameters(ignore=["model_class", "pos_weight"])  # callable unpicklable under weights_only; pass at load
        self.model = model_class(**(model_kwargs or {}))
        self.register_buffer("pos_weight", pos_weight)  # buffer: moves to device, saved in checkpoint
        self.train_map = mean_ap()
        self.val_map = mean_ap()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, metric):
        x, y = batch
        logits = self(x)
        metric.update(torch.sigmoid(logits.float()), y.int())
        # one independent sigmoid per class
        return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_map)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_map", self.train_map.compute(), prog_bar=True)
        self.train_map.reset()

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, self.val_map)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_map", self.val_map.compute(), prog_bar=True)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=self.hparams.final_lr
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


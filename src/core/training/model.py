"""Lightning wrapper of any CNN with one logit per class: Adam, cosine schedule per batch, BCE, mAP.

Networks: YAML `model.net` block (`class_path`, `init_args`) names `nn.Module` class or function returning one:
`timm_net`, `alexnet_imagenet`, own nets in `resnet_paper`.
"""

import importlib
import logging
import warnings

import lightning.pytorch as pl
import timm
import torch
import torchvision
from torch import nn

from ..data.dataset import sample_weights
from .config import CLASSES

warnings.filterwarnings(
    "ignore", message="Attribute 'net' is an instance"
)  # kept on purpose: checkpoint rebuilds `net` from its YAML block


def timm_net(model_name: str, pretrained: bool = True) -> nn.Module:
    """timm model `model_name` with one logit per class; typed wrapper: `timm.create_model` takes `**kwargs`, which
    LightningCLI does not accept."""
    return timm.create_model(
        model_name, pretrained=pretrained, num_classes=len(CLASSES)
    )


def alexnet_imagenet(num_classes: int = len(CLASSES)) -> nn.Module:
    """torchvision AlexNet with ImageNet-1k weights (no timm version); last layer replaced for `num_classes`."""
    model = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    )
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    return model


class LitCNNMultiLabel(pl.LightningModule):
    """`net`: CNN with one logit per class. Adam (`lr`), `CosineAnnealingLR` to `eta_min` stepped per batch, BCE with
    `pos_weight` of datamodule training rows and sampler, ratio to `pos_weight_power` (set at fit start, saved in checkpoint).
    Steps return `loss`, `probs`, `labels`: metrics in callbacks (`core.training.train.MeanAP`). Trained one: `load`."""

    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        eta_min: float = 1e-5,
        pos_weight_power: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()  # under LightningCLI: YAML block of `net`, so checkpoint rebuilds it
        self.model = net
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones(len(CLASSES)))

    @classmethod
    def load(cls, ckpt, map_location=None):
        """Trained model of checkpoint `ckpt`. Checkpoints before LightningCLI: network from their `model` block
        (`_target_` plus kwargs), weights from state dict."""
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)

        if "model" not in checkpoint["hyper_parameters"]:
            return cls.load_from_checkpoint(ckpt, map_location=map_location)

        block = dict(checkpoint["hyper_parameters"]["model"])
        module, _, name = block.pop("_target_").rpartition(".")
        module = (
            "core.training.model" if module.startswith("core.") else module
        )  # project nets moved here since
        net = getattr(importlib.import_module(module), name)(
            **block, num_classes=len(CLASSES)
        )
        model = cls(net)
        model.load_state_dict(checkpoint["state_dict"])

        return model.to(map_location) if map_location else model

    def pos_weight(self, dm):
        """BCE `pos_weight` of training rows of datamodule `dm`: ((N - n_c) / n_c) ** `pos_weight_power`, N and n_c
        counted over sampler draws."""
        train = dm.df[dm.df["fold"] != dm.fold]
        counts = sample_weights(train, dm.oversample, dm.downsample)
        n_pos = counts @ train[CLASSES].to_numpy()

        return torch.tensor(
            ((counts.sum() - n_pos) / n_pos) ** self.hparams.pos_weight_power,
            dtype=torch.float32,
        )

    def setup(self, stage):
        if stage == "fit":
            self.loss.pos_weight.copy_(self.pos_weight(self.trainer.datamodule))

    def predict_probs(self, loader):
        """Sigmoid probabilities and labels of every row of `loader`, as numpy arrays (`trainer.predict`)."""
        logger = logging.getLogger("lightning.pytorch")
        level = logger.level
        logger.setLevel(logging.WARNING)  # no device info lines per call

        try:
            trainer = pl.Trainer(
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
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

    def _step(self, batch):
        x, y = batch
        logits = self(x)

        return {
            "loss": self.loss(logits, y),
            "probs": torch.sigmoid(logits.float()).detach(),
            "labels": y,
        }

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)

        return out

    def validation_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log("val_loss", out["loss"], on_epoch=True, prog_bar=True)

        return out

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

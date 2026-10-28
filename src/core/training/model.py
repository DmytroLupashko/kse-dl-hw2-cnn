"""Lightning wrapper of any CNN with one logit per class: Adam, cosine schedule per batch, BCE, mAP.

Networks: YAML `model.net` block (`class_path`, `init_args`) names `nn.Module` class or function returning one:
`timm_net`, `alexnet_imagenet`, own classes below.
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


class ConvBlock(nn.Module):
    """Conv, BatchNorm (`use_batchnorm`, else Identity), ReLU, then 3x3 stride-2 max pool when `pool`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        pool=False,
        use_batchnorm=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class ResnetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))


# I did
class ResnetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, initial=False):
        super().__init__()
        conv1_stride = 2 if initial else 1
        self.conv1 = ResnetConv(
            in_channels,
            out_channels,
            3,
            stride=conv1_stride,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ResnetConv(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcutConv = (
            ResnetConv(
                in_channels,
                out_channels,
                1,
                stride=conv1_stride,
                use_batchnorm=use_batchnorm,
            )
            if initial
            else nn.Identity()
        )

    def forward(self, x):
        y1 = self.relu1(self.conv1(x))
        y2 = self.conv2(y1)
        x1 = self.shortcutConv(x)

        return self.relu2(y2 + x1)


# I did, unfinished
class ResnetFromPaper(nn.Module):
    # By default Paper uses batchnorm
    # It does not use dropout
    def __init__(self, num_classes, use_batchnorm=True):
        super().__init__()

        self.inputConv = ConvBlock(
            3, 64, 7, stride=2, padding=3, use_batchnorm=use_batchnorm
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnetBlock64_1 = ResnetConvBlock(64, 64, use_batchnorm=use_batchnorm)
        self.resnetBlock64_2 = ResnetConvBlock(64, 64, use_batchnorm=use_batchnorm)
        self.resnetBlock64_3 = ResnetConvBlock(64, 64, use_batchnorm=use_batchnorm)

        self.resnetBlock128_1 = ResnetConvBlock(
            64, 128, use_batchnorm=use_batchnorm, initial=True
        )
        self.resnetBlock128_2 = ResnetConvBlock(128, 128, use_batchnorm=use_batchnorm)
        self.resnetBlock128_3 = ResnetConvBlock(128, 128, use_batchnorm=use_batchnorm)
        self.resnetBlock128_4 = ResnetConvBlock(128, 128, use_batchnorm=use_batchnorm)

        self.resnetBlock256_1 = ResnetConvBlock(
            128, 256, use_batchnorm=use_batchnorm, initial=True
        )
        self.resnetBlock256_2 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_3 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_4 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_5 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_6 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)

        self.resnetBlock512_1 = ResnetConvBlock(
            256, 512, use_batchnorm=use_batchnorm, initial=True
        )
        self.resnetBlock512_2 = ResnetConvBlock(512, 512, use_batchnorm=use_batchnorm)
        self.resnetBlock512_3 = ResnetConvBlock(512, 512, use_batchnorm=use_batchnorm)

        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.inputConv(x)
        x = self.pool1(x)

        x = self.resnetBlock64_1(x)
        x = self.resnetBlock64_2(x)
        x = self.resnetBlock64_3(x)

        x = self.resnetBlock128_1(x)
        x = self.resnetBlock128_2(x)
        x = self.resnetBlock128_3(x)
        x = self.resnetBlock128_4(x)

        x = self.resnetBlock256_1(x)
        x = self.resnetBlock256_2(x)
        x = self.resnetBlock256_3(x)
        x = self.resnetBlock256_4(x)
        x = self.resnetBlock256_5(x)
        x = self.resnetBlock256_6(x)

        x = self.resnetBlock512_1(x)
        x = self.resnetBlock512_2(x)
        x = self.resnetBlock512_3(x)

        x = self.avgPool(x)
        x = torch.flatten(x, 1)

        return self.head(x)


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

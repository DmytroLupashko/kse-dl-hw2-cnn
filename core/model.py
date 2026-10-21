"""Lightning wrapper of any CNN with `num_classes` logits: weighted BCE loss, mAP, Adam + cosine schedule."""
import lightning.pytorch as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from tqdm import tqdm

from .config import CLASSES


def alexnet_imagenet(num_classes):
    """torchvision AlexNet with ImageNet-1k weights (no timm version); last layer replaced for `num_classes`."""
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


TORCHVISION_MODELS = {"alexnet": alexnet_imagenet}  # name: builder taking `num_classes`, ImageNet weights

class ConvBlock(nn.Module):
    """Conv, BatchNorm (`use_batchnorm`, else Identity), ReLU, then 3x3 stride-2 max pool when `pool`."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool=False, use_batchnorm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class DenseBlock(nn.Module):
    """Linear, ReLU, Dropout."""

    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.fc(x)))


class AlexNet(nn.Module):
    """AlexNet of Lecture 1 (`Old_Good_CNNs`), trained from scratch: five `ConvBlock`s, two `DenseBlock`s, linear head.
    Adaptive pool to 6x6 before head: no-op at lecture's 227 px input, lets 224 px work; inputs below ~200 px too small.
    Lecture's hooks on `conv1`..`conv5` go on `conv1.conv`..`conv5.conv`."""

    def __init__(self, num_classes, use_batchnorm=False, dropout=0.6):
        super().__init__()
        self.conv1 = ConvBlock(3, 96, kernel_size=11, stride=4, pool=True, use_batchnorm=use_batchnorm)
        self.conv2 = ConvBlock(96, 256, kernel_size=5, padding=2, pool=True, use_batchnorm=use_batchnorm)
        self.conv3 = ConvBlock(256, 384, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv4 = ConvBlock(384, 384, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv5 = ConvBlock(384, 256, kernel_size=3, padding=1, pool=True, use_batchnorm=use_batchnorm)
        self.pool = nn.AdaptiveAvgPool2d(6)
        self.fc1 = DenseBlock(256 * 6 * 6, 4096, dropout)
        self.fc2 = DenseBlock(4096, 4096, dropout)
        self.head = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(self.pool(x), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.head(x)


# I did
class ResnetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3, stride=1, padding=0)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, stride=1, padding=0)
        
    def forward(self, x):
        return self.conv2(self.conv1(x)) + x

# I did, unfinished
class ResnetFromPaper(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    
    self.inputConv = ConvBlock(num_classes, 64, )
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    self.resnetBlock64_1 = ResnetConvBlock(64, 64)
    self.resnetBlock64_1 = ResnetConvBlock(64, 64)
    self.resnetBlock64_1 = ResnetConvBlock(64, 64)
    
    self.resnetBlock128_1 = ResnetConvBlock(64, 128)
    self.resnetBlock128_2 = ResnetConvBlock(128, 128)
    self.resnetBlock128_3 = ResnetConvBlock(128, 128)
    self.resnetBlock128_4 = ResnetConvBlock(128, 256)
    
    self.resnetBlock128_1 = ResnetConvBlock(128, 256)
    self.resnetBlock128_2 = ResnetConvBlock(128, 256)
    self.resnetBlock128_3 = ResnetConvBlock(128, 256)
    self.resnetBlock128_4 = ResnetConvBlock(128, 256)
    self.resnetBlock128_5 = ResnetConvBlock(128, 256)
    self.resnetBlock128_6 = ResnetConvBlock(128, 256)
    
    self.resnetBlock128_1 = ResnetConvBlock(128, 256)
    self.resnetBlock128_2 = ResnetConvBlock(128, 256)
    self.resnetBlock128_3 = ResnetConvBlock(128, 256)
    self.resnetBlock128_4 = ResnetConvBlock(128, 256)

class LitCNNMultiLabel(pl.LightningModule):
    """`pos_weight`: tensor from `KaggleDataModule.pos_weight`; None means plain BCE.
    New model: `build`; trained one: `load` (`load_from_checkpoint` needs same `model_class` and `pos_weight` passed again)."""

    def __init__(self, model_class=timm.create_model, model_kwargs=None, initial_lr=1e-3, final_lr=1e-5,
                 pos_weight=None):
        super().__init__()
        self.save_hyperparameters(ignore=["model_class", "pos_weight"])  # callable unpicklable under weights_only; pass at load
        self.model = model_class(**(model_kwargs or {}))
        self.register_buffer("pos_weight", pos_weight)  # buffer: moves to device, saved in checkpoint
        self.train_map = self._mean_ap()
        self.val_map = self._mean_ap()

    @staticmethod
    def _mean_ap():
        return torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(CLASSES), average="macro")

    @staticmethod
    def _backbone(timm_model, torchvision_model):
        """(model_class, model_kwargs) of exactly one of `timm_model` (pretrained) or `torchvision_model` (`TORCHVISION_MODELS` key)."""
        if timm_model is not None:
            return timm.create_model, {"model_name": timm_model, "pretrained": True, "num_classes": len(CLASSES)}
        return TORCHVISION_MODELS[torchvision_model], {"num_classes": len(CLASSES)}

    @classmethod
    def build(cls, timm_model=None, torchvision_model=None, initial_lr=1e-3, final_lr=1e-5, pos_weight=None):
        """New model on backbone `timm_model` or `torchvision_model`."""
        model_class, model_kwargs = cls._backbone(timm_model, torchvision_model)
        return cls(
            model_class=model_class, model_kwargs=model_kwargs, initial_lr=initial_lr, final_lr=final_lr, pos_weight=pos_weight
        )

    @classmethod
    def load(cls, ckpt, timm_model=None, torchvision_model=None, map_location=None):
        """Trained model from checkpoint `ckpt`; backbone args as at `build`.
        `pos_weight` buffer restored from checkpoint; passed only for its shape."""
        model_class, model_kwargs = cls._backbone(timm_model, torchvision_model)
        return cls.load_from_checkpoint(
            ckpt, map_location=map_location, model_class=model_class, model_kwargs=model_kwargs,
            pos_weight=torch.ones(len(CLASSES)),
        )

    @torch.no_grad()
    def predict_probs(self, loader):
        """Sigmoid probabilities and labels of every row of `loader`, as numpy arrays."""
        self.eval()
        probs, targets = [], []
        for x, y in tqdm(loader, desc="predict"):
            probs.append(torch.sigmoid(self(x.to(self.device)).float()).cpu())
            targets.append(y)
        return torch.cat(probs).numpy(), torch.cat(targets).numpy()

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

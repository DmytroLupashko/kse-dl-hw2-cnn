"""WS-DAN (Hu et al. 2019, arXiv 1901.09891) for multilabel: attention maps, bilinear attention pooling, center loss,
attention cropping and dropping in training, attention crop refinement at prediction.
Details follow official code (github.com/GuYuc/WS-DAN.PyTorch)."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LitCNNMultiLabel


class WSDAN(nn.Module):
    """timm backbone last feature map, `num_maps` attention maps (1x1 conv), bilinear attention pooling, linear layer.
    Returns (logits, part features (B, num_maps * C), attention maps (B, num_maps, h, w))."""

    def __init__(self, model_name, num_classes, num_maps=32, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[-1])
        channels = self.backbone.feature_info.channels()[-1]
        self.attention = nn.Sequential(nn.Conv2d(channels, num_maps, 1, bias=False), nn.BatchNorm2d(num_maps), nn.ReLU())
        self.fc = nn.Linear(num_maps * channels, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)[-1]
        maps = self.attention(features)
        # bilinear attention pooling: feature map times each attention map, averaged over pixels
        parts = torch.einsum("bmhw,bchw->bmc", maps, features) / (features.shape[2] * features.shape[3])
        parts = torch.sign(parts) * torch.sqrt(parts.abs() + 1e-12)
        parts = F.normalize(parts.flatten(1), dim=-1)
        return self.fc(parts * 100), parts, maps


def sample_maps(maps, n=2):
    """Per image `n` distinct attention maps, drawn with prob proportional to sqrt of map sum. Shape (B, n, h, w)."""
    weights = maps.sum(dim=(2, 3)).sqrt() + 1e-12
    idx = torch.multinomial(weights, n)
    return maps[torch.arange(len(maps), device=maps.device)[:, None], idx]


def _scaled(maps, x):
    """(B, h, w) maps upsampled to image size of `x`, each divided by own max."""
    maps = F.interpolate(maps[:, None], size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
    return maps / (maps.amax(dim=(1, 2), keepdim=True) + 1e-12)


def attention_crop(x, maps, thresholds, padding):
    """Per image: box of pixels with map >= threshold (fraction of map max), widened by `padding` of image side,
    cut out, resized to input size. Dead map (all zero): whole image."""
    h, w = x.shape[2:]
    crops = []
    for img, m, t in zip(x, _scaled(maps, x), thresholds):
        ys, xs = torch.nonzero(m >= t, as_tuple=True)
        if len(ys) and m.max() > 0:
            y0, y1 = max(int(ys.min()) - int(padding * h), 0), min(int(ys.max()) + int(padding * h) + 1, h)
            x0, x1 = max(int(xs.min()) - int(padding * w), 0), min(int(xs.max()) + int(padding * w) + 1, w)
            img = img[:, y0:y1, x0:x1]
        crops.append(F.interpolate(img[None], size=(h, w), mode="bilinear", align_corners=False))
    return torch.cat(crops)


def attention_drop(x, maps, thresholds):
    """Per image: pixels with map >= threshold (fraction of map max) set to 0 (gray after `Normalize`)."""
    keep = _scaled(maps, x) < thresholds[:, None, None]
    return x * keep[:, None].to(x.dtype)


class LitWSDAN(LitCNNMultiLabel):
    """`LitCNNMultiLabel` with `WSDAN` network. Training loss: mean weighted BCE of raw, attention-cropped and
    attention-dropped image (labels kept) plus center loss of part features. Prediction (`forward`): mean of
    probabilities on full image and on crop of mean attention map, returned as logits.
    Center loss: one shared center per part, not per class (multilabel image has several classes)."""

    def __init__(self, model_class=WSDAN, model_kwargs=None, initial_lr=1e-3, final_lr=1e-5, pos_weight=None,
                 crop_thresholds=(0.4, 0.6), drop_thresholds=(0.2, 0.5), center_beta=0.05):
        super().__init__(model_class, model_kwargs, initial_lr, final_lr, pos_weight)
        self.register_buffer("centers", torch.zeros(self.model.fc.in_features))  # moving average of part features

    def forward(self, x):
        logits, _, maps = self.model(x)
        crop = attention_crop(x, maps.mean(1), torch.full((len(x),), 0.1), padding=0.05)
        probs = (torch.sigmoid(logits.float()) + torch.sigmoid(self.model(crop)[0].float())) / 2
        return torch.logit(probs, eps=1e-6)

    def training_step(self, batch, batch_idx):
        x, y = batch
        hp = self.hparams
        logits, parts, maps = self.model(x)
        center_loss = (parts - self.centers).pow(2).sum(1).mean()
        with torch.no_grad():
            self.centers += hp.center_beta * (parts.mean(0) - self.centers)
        crop_map, drop_map = sample_maps(maps.detach()).unbind(1)
        x_crop = attention_crop(x, crop_map, torch.empty(len(x)).uniform_(*hp.crop_thresholds), padding=0.1)
        x_drop = attention_drop(x, drop_map, torch.empty(len(x), device=x.device).uniform_(*hp.drop_thresholds))
        bce = [
            F.binary_cross_entropy_with_logits(out, y, pos_weight=self.pos_weight)
            for out in (logits, self.model(x_crop)[0], self.model(x_drop)[0])
        ]
        loss = sum(bce) / 3 + center_loss
        self.train_map.update(torch.sigmoid(logits.float()), y.int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("center_loss", center_loss, on_step=False, on_epoch=True)
        return loss

"""Grad-CAM of fold models on their out-of-fold rows: per class, figures of correct and incorrect examples."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from .config import CLASSES, MEAN, TRAIN_DIR
from .cv import DEVICE, CVResult
from .data import read_train, val_tfms
from .predict import best_f1


def grad_cam(model, layer, x, class_idx):
    """Grad-CAM of logit `class_idx` at output of `layer` (module of `model`), one map per image of batch `x`.
    Maps upsampled to input size, scaled to max 1 per image, as numpy array (batch, height, width)."""
    acts = []
    handle = layer.register_forward_hook(lambda _module, _inputs, out: acts.append(out))
    try:
        model.eval()
        score = model(x)[:, class_idx].sum()
    finally:
        handle.remove()
    (grads,) = torch.autograd.grad(score, acts[0])  # no parameter grads accumulated
    cam = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * acts[0]).sum(dim=1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    return cam[:, 0].detach().cpu().numpy()


def erase(image, box):
    """Copy of HWC uint8 `image` with `box` (x_min, y_min, x_max, y_max in [0, 1]) filled with normalization mean."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    image = image.copy()
    image[int(y0 * h) : int(np.ceil(y1 * h)), int(x0 * w) : int(np.ceil(x1 * w))] = np.array(MEAN) * 255
    return image


def random_box(box, rng, tries=100):
    """Box of same size as `box` at random place not overlapping it; last try when none fits."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    for _ in range(tries):
        nx, ny = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
        if nx + w <= x0 or nx >= x1 or ny + h <= y0 or ny >= y1:
            break
    return nx, ny, nx + w, ny + h


def last_feature_layer(model):
    """Module whose output is last feature map: ResNet `layer4`, timm EfficientNet `bn2` (`conv_head` + BN + activation),
    RegNet `s4`, ConvNeXt `stages.3`."""
    names = dict(model.named_modules())
    for name in ("layer4", "bn2", "s4", "stages.3"):
        if name in names:
            return names[name]
    raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")


class GradCamFigures:
    """Fold models of `config` (Grad-CAM at `last_feature_layer`), out-of-fold probs, best-F1 threshold per class."""

    def __init__(self, config):
        self.config = config
        result = CVResult.load(config)
        self.oof = result.oof  # index = row of read_train()
        self.image_ids = read_train()["image_id"]
        self.thresholds = best_f1(result.labels, result.probs)["threshold"]
        self.models = {fold: config.fold_model(fold).model for fold in config.folds}

    def pick(self, cls, n):
        """{"correct": n most confident true positives, "incorrect": n // 2 most confident false positives
        plus n // 2 most confident false negatives}, as (row, "TP" | "FP" | "FN") pairs."""
        p, y = self.oof[f"p_{cls}"], self.oof[cls] == 1
        pred = p >= self.thresholds[cls]
        return {
            "correct": [(i, "TP") for i in p[y & pred].nlargest(n).index],
            "incorrect": [(i, "FP") for i in p[~y & pred].nlargest(n // 2).index]
            + [(i, "FN") for i in p[y & ~pred].nsmallest(n // 2).index],
        }

    def load(self, idx, box=None):
        """Training image of row `idx` as HWC uint8, `box` erased (`erase`) when given."""
        image = np.array(Image.open(TRAIN_DIR / f"{self.image_ids[idx]}.jpg").convert("RGB"))
        return image if box is None else erase(image, box)

    def probs(self, rows, cls, boxes=None, batch=64):
        """Prob of `cls` per row of `rows` by model that held row out; `boxes`: {row: box} erased first."""
        boxes = boxes or {}
        tfms = val_tfms(self.config.img_size)
        out = {}
        for fold, part in pd.Series(rows, index=rows).groupby(self.oof.loc[rows, "fold"].to_numpy()):
            model = self.models[fold].eval()
            for start in range(0, len(part), batch):
                idx = part.iloc[start : start + batch].to_numpy()
                x = torch.stack([tfms(image=self.load(i, boxes.get(i)))["image"] for i in idx]).to(DEVICE)
                with torch.no_grad():
                    p = torch.sigmoid(model(x)[:, CLASSES.index(cls)]).cpu().numpy()
                out.update(zip(idx, p))
        return pd.Series(out).reindex(rows)

    def image_and_cam(self, idx, cls, fold=None, box=None):
        """(resized image, Grad-CAM, prob of `cls`) by model of `fold`; default: model that held image out.
        `box` erased first when given."""
        size = self.config.img_size
        image = self.load(idx, box)
        x = val_tfms(size)(image=image)["image"][None].to(DEVICE)
        model = self.models[self.oof.loc[idx, "fold"] if fold is None else fold]
        cam = grad_cam(model, last_feature_layer(model), x, CLASSES.index(cls))[0]
        with torch.no_grad():
            prob = torch.sigmoid(model(x)[0, CLASSES.index(cls)]).item()
        return np.array(Image.fromarray(image).resize((size, size))), cam, prob

    def plot(self, cls, examples):
        """One column per example: original on top, Grad-CAM overlay of `cls` logit below."""
        fig, axes = plt.subplots(2, len(examples), figsize=(3.6 * len(examples), 7), squeeze=False)
        for (idx, kind), ax_img, ax_cam in zip(examples, axes[0], axes[1]):
            shown, cam, _ = self.image_and_cam(idx, cls)
            row = self.oof.loc[idx]
            true = [c for c in CLASSES if row[c] == 1] or ["none"]
            pred = [c for c in CLASSES if row[f"p_{c}"] >= self.thresholds[c]] or ["none"]
            ax_img.imshow(shown)
            ax_img.set_title(f"Original\n{', '.join(true)}", fontsize=9)
            ax_cam.imshow(shown)
            ax_cam.imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.5)
            # lecture style: green when prediction of this class right, red otherwise
            ax_cam.set_title(
                f"Overlay · {kind}, p {row[f'p_{cls}']:.3f}\nPred: {', '.join(pred)}",
                fontsize=9,
                color="green" if kind == "TP" else "red",
            )
        for ax in axes.flat:
            ax.set_axis_off()
        fig.tight_layout(w_pad=3, h_pad=2)  # titles of neighbor images must not touch
        return fig

    def positive_rows(self, cls, n=None, seed=None):
        """Rows of labeled positives of `cls` (first `n`, or with `seed`: random `n`), sorted by out-of-fold prob, high first."""
        probs = self.oof.loc[self.oof[cls] == 1, f"p_{cls}"]
        if seed is not None and n is not None:
            probs = probs.sample(min(n, len(probs)), random_state=seed)
        return probs.sort_values(ascending=False).index[:n]

    def positives(self, cls, train=False, n_cols=8, n=None, seed=None):
        """All labeled positives of `cls` (first `n`, or with `seed`: random `n`), sorted by out-of-fold prob, high first;
        per image original on top, Grad-CAM overlay below. Maps by model that held image out, or `train`: by one that
        trained on it (next fold)."""
        rows = self.positive_rows(cls, n, seed)
        folds = self.config.folds
        n_rows = -(-len(rows) // n_cols)
        fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(2.2 * n_cols, 4.8 * n_rows), squeeze=False)
        for k, idx in enumerate(rows):
            fold = self.oof.loc[idx, "fold"]
            if train:
                fold = folds[(folds.index(fold) + 1) % len(folds)]
            shown, cam, prob = self.image_and_cam(idx, cls, fold)
            r, c = divmod(k, n_cols)
            axes[2 * r, c].imshow(shown)
            axes[2 * r + 1, c].imshow(shown)
            axes[2 * r + 1, c].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.5)
            # green when this model predicts `cls` at its threshold, red when it misses
            axes[2 * r + 1, c].set_title(
                f"fold {fold} · p {prob:.3f}", fontsize=8, color="green" if prob >= self.thresholds[cls] else "red"
            )
        for ax in axes.flat:
            ax.set_axis_off()
        split = "train (model trained on image)" if train else "validation (model held image out)"
        fig.suptitle(f"{self.config.name} · {cls} · {split} · {len(rows)} labeled positives", color="#52514e")
        fig.tight_layout()
        return fig

    def figures(self, classes, n):
        """{(class, "correct" | "incorrect"): figure}; outcomes without examples skipped."""
        return {
            (cls, outcome): self.plot(cls, examples)
            for cls in classes
            for outcome, examples in self.pick(cls, n).items()
            if examples
        }


def compare(cams, cls, rows, n_cols=8, boxes=None):
    """Grad-CAM of several models, {name: GradCamFigures}, on same validation `rows` (labeled positives of `cls`):
    per image original on top, then one overlay per model, each by its fold model that held image out.
    `boxes`: {row: box} erased in every image first."""
    boxes = boxes or {}
    names = list(cams)
    per = 1 + len(names)
    n_groups = -(-len(rows) // n_cols)
    fig, axes = plt.subplots(per * n_groups, n_cols, figsize=(2.2 * n_cols, 2.4 * per * n_groups), squeeze=False)
    for k, idx in enumerate(rows):
        g, c = divmod(k, n_cols)
        for m, name in enumerate(names):
            shown, cam, prob = cams[name].image_and_cam(idx, cls, box=boxes.get(idx))
            if m == 0:
                axes[per * g, c].imshow(shown)
            ax = axes[per * g + 1 + m, c]
            ax.imshow(shown)
            ax.imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.5)
            # green when this model predicts `cls` at its threshold, red when it misses
            ax.set_title(f"{name} · p {prob:.3f}", fontsize=8, color="green" if prob >= cams[name].thresholds[cls] else "red")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle(f"{cls} · validation (model held image out) · {len(rows)} labeled positives", color="#52514e")
    fig.tight_layout()
    return fig

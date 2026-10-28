"""Grad-CAM of fold models on their out-of-fold rows: per class, figures of correct and incorrect examples."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ..data.augment import Validation
from ..data.dataset import read_train
from ..training.config import CLASSES, DEVICE, TRAIN_DIR
from ..training.cv import CVResult
from .thresholds import best_f1


def overlay(ax, image, cam, title, hit, fontsize=8):
    """Grad-CAM `cam` over `image` on `ax`; lecture style title: green when prediction right (`hit`), red otherwise."""
    ax.imshow(image)
    ax.imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=0.5)
    ax.set_title(title, fontsize=fontsize, color="green" if hit else "red")


class GradCam:
    """Grad-CAM (`pytorch_grad_cam.GradCAM`) of `model` at its last feature map; call `(x, class_idx)` -> one map per
    image of batch `x`, upsampled to input size, min-max scaled to [0, 1] per image, numpy array (batch, height, width)."""

    def __init__(self, model):
        self.model = model
        names = dict(model.named_modules())
        # last feature map: ResNet, timm EfficientNet (`conv_head` + BN + activation), RegNet, ConvNeXt
        layer = next(
            (n for n in ("layer4", "bn2", "s4", "stages.3") if n in names), None
        )

        if layer is None:
            raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")

        self.layer = names[layer]

    def __call__(self, x, class_idx):
        self.model.eval()

        # hooks removed on exit: other forward passes keep no activations
        with GradCAM(self.model, target_layers=[self.layer]) as cam:
            return cam(x, targets=[ClassifierOutputTarget(class_idx)] * len(x))


class GradCamFigures:
    """Fold models of model config `path` (`GradCam` each), out-of-fold probs, best-F1 threshold per class."""

    def __init__(self, path):
        result = CVResult.load(path)
        self.config, self.folds = result.config, result.folds
        self.oof = result.oof  # index = row of read_train()
        self.image_ids = read_train()["image_id"]
        self.thresholds = best_f1(result.labels, result.probs, CLASSES)["threshold"]
        self.cams = {
            fold: GradCam(result.fold_model(fold).model) for fold in result.folds
        }
        self.tfms = Validation(result.config["data"]["img_size"]).compose()

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

    def load(self, idx, edit=None):
        """Training image of row `idx` as HWC uint8, `edit` (HWC uint8 -> HWC uint8) applied when given."""
        image = np.array(
            Image.open(TRAIN_DIR / f"{self.image_ids[idx]}.jpg").convert("RGB")
        )

        return image if edit is None else edit(image)

    def probs(self, rows, cls, edits=None):
        """Prob of `cls` per row of `rows` by model that held row out; `edits`: {row: `edit` of `load`} applied first."""
        edits = edits or {}
        out = {}

        for fold, part in pd.Series(rows, index=rows).groupby(
            self.oof.loc[rows, "fold"].to_numpy()
        ):
            model = self.cams[fold].model.eval()

            for start in range(0, len(part), 64):
                idx = part.iloc[start : start + 64].to_numpy()
                x = torch.stack(
                    [self.tfms(image=self.load(i, edits.get(i)))["image"] for i in idx]
                ).to(DEVICE)

                with torch.no_grad():
                    p = torch.sigmoid(model(x)[:, CLASSES.index(cls)]).cpu().numpy()

                out.update(zip(idx, p))

        return pd.Series(out).reindex(rows)

    def image_and_cam(self, idx, cls, fold=None, edit=None):
        """(resized image, Grad-CAM, prob of `cls`) by model of `fold`; default: model that held image out.
        `edit` of `load` applied first when given."""
        size = self.config["data"]["img_size"]
        image = self.load(idx, edit)
        x = self.tfms(image=image)["image"][None].to(DEVICE)
        fold = self.oof.loc[idx, "fold"] if fold is None else fold
        cam = self.cams[fold](x, CLASSES.index(cls))[0]

        with torch.no_grad():
            prob = torch.sigmoid(self.cams[fold].model(x)[0, CLASSES.index(cls)]).item()

        return np.array(Image.fromarray(image).resize((size, size))), cam, prob

    def plot(self, cls, examples):
        """One column per example: original on top, Grad-CAM overlay of `cls` logit below."""
        fig, axes = plt.subplots(
            2, len(examples), figsize=(3.6 * len(examples), 7), squeeze=False
        )

        for (idx, kind), ax_img, ax_cam in zip(examples, axes[0], axes[1]):
            shown, cam, _ = self.image_and_cam(idx, cls)
            row = self.oof.loc[idx]
            true = [c for c in CLASSES if row[c] == 1] or ["none"]
            pred = [c for c in CLASSES if row[f"p_{c}"] >= self.thresholds[c]] or [
                "none"
            ]
            ax_img.imshow(shown)
            ax_img.set_title(f"Original\n{', '.join(true)}", fontsize=9)
            overlay(
                ax_cam,
                shown,
                cam,
                f"Overlay · {kind}, p {row[f'p_{cls}']:.3f}\nPred: {', '.join(pred)}",
                kind == "TP",
                9,
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

    def positives(self, cls, train=False, n=None, seed=None):
        """All labeled positives of `cls` (first `n`, or with `seed`: random `n`), sorted by out-of-fold prob, high first;
        `compare` with this model alone."""
        return self.compare(
            {self.config["name"]: self},
            cls,
            self.positive_rows(cls, n, seed),
            train=train,
        )

    def figures(self, classes, n):
        """{(class, "correct" | "incorrect"): figure}; outcomes without examples skipped."""
        return {
            (cls, outcome): self.plot(cls, examples)
            for cls in classes
            for outcome, examples in self.pick(cls, n).items()
            if examples
        }

    @staticmethod
    def compare(cams, cls, rows, edits=None, train=False):
        """Grad-CAM of several models, {name: GradCamFigures}, on same `rows` (labeled positives of `cls`):
        per image original on top, then one overlay per model, each by its fold model that held image out, or `train`:
        by one that trained on it (next fold). `edits`: {row: `edit` of `load`} applied in every image first."""
        edits = edits or {}
        names = list(cams)
        per, n_cols = 1 + len(names), 8
        n_groups = -(-len(rows) // n_cols)
        fig, axes = plt.subplots(
            per * n_groups,
            n_cols,
            figsize=(2.2 * n_cols, 2.4 * per * n_groups),
            squeeze=False,
        )

        for k, idx in enumerate(rows):
            g, c = divmod(k, n_cols)

            for m, name in enumerate(names):
                folds = cams[name].folds
                fold = cams[name].oof.loc[idx, "fold"]

                if train:
                    fold = folds[(folds.index(fold) + 1) % len(folds)]

                shown, cam, prob = cams[name].image_and_cam(
                    idx, cls, fold, edits.get(idx)
                )

                if m == 0:
                    axes[per * g, c].imshow(shown)

                title = f"{name} · fold {fold} · p {prob:.3f}"
                overlay(
                    axes[per * g + 1 + m, c],
                    shown,
                    cam,
                    title,
                    prob >= cams[name].thresholds[cls],
                )

        for ax in axes.flat:
            ax.set_axis_off()

        split = (
            "train (model trained on image)"
            if train
            else "validation (model held image out)"
        )
        fig.suptitle(
            f"{cls} · {split} · {len(rows)} labeled positives", color="#52514e"
        )
        fig.tight_layout()

        return fig

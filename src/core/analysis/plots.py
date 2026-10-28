"""Notebook figures: shared style (dataviz reference palette, light mode), saving, PR curves, image previews."""

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

from ..training.config import CLASSES, MEAN, STD
from .metrics import ap, pr_curve
from .thresholds import best_f1

INK, PAPER, GRID = "#52514e", "#fcfcfb", "#e1e0d9"
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#7c5cd6",
]  # series colors in order


def use_style():
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titlesize": "medium",
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "legend.frameon": False,
            "legend.fontsize": "small",
        }
    )


def save_figures(figures, folder, dpi=120, **savefig_kw):
    """{file name: figure} saved to `folder`."""
    for fname, fig in figures.items():
        fig.savefig(folder / fname, dpi=dpi, **savefig_kw)


def to_image(x):
    """Normalized CHW tensor back to HWC image in [0, 1]."""
    return (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)


def pr_curves(results, classes=CLASSES, colors=None, title="Out-of-fold rows"):
    """PR curve of `classes` on pooled OOF rows of {name: `CVResult`}, one panel per class, one color per model
    (`colors` {name: color}, default `PALETTE` order); AP in legend, dot at best-F1 threshold.
    """
    colors = colors or dict(zip(results, PALETTE))
    aps = {name: ap(r.labels, r.probs, CLASSES) for name, r in results.items()}
    bests = {name: best_f1(r.labels, r.probs, CLASSES) for name, r in results.items()}
    fig, axes = plt.subplots(
        1, len(classes), figsize=(3.4 * len(classes), 3.6), sharey=True, squeeze=False
    )

    for cls, ax in zip(classes, axes[0]):
        for name, r in results.items():
            color = colors[name]
            y, p = r.oof[cls], r.oof[f"p_{cls}"]
            precision, recall, _, _ = pr_curve(y, p)
            ax.plot(
                recall,
                precision,
                color=color,
                lw=2,
                drawstyle="steps-post",
                label=f"{name}, AP {aps[name][cls]:.3f}",
            )
            best = bests[name].loc[cls]
            ax.plot(
                best["recall"],
                best["precision"],
                "o",
                ms=7,
                color=color,
                mec=PAPER,
                mew=2,
            )

        ax.set(
            title=f"{cls} ({int(y.sum())} positives)",
            xlabel="recall",
            xlim=(0, 1.02),
            ylim=(0, 1.05),
        )
        ax.legend(loc="lower left", alignment="left")

    axes[0, 0].set_ylabel("precision")
    fig.suptitle(f"{title}; dot = best-F1 threshold", color=INK)
    fig.tight_layout()

    return fig


def preview(tfms):
    """Recipe Compose `tfms` without Normalize, ToTensorV2, box handling kept: draws as uint8 images to show."""
    bbox = tfms.processors.get("bboxes")

    return A.Compose(tfms.transforms[:-2], bbox_params=bbox.params if bbox else None)


def draw_boxes(ax, image, bboxes, labels, color, ls="-"):
    """Normalized `bboxes` outlined on `ax` over `image`, each tagged with its label at top left."""
    h, w = image.shape[:2]

    for (x0, y0, x1, y1), cls in zip(bboxes, labels):
        ax.add_patch(
            plt.Rectangle(
                (x0 * w, y0 * h),
                (x1 - x0) * w,
                (y1 - y0) * h,
                fill=False,
                ec=color,
                lw=1.5,
                ls=ls,
            )
        )
        ax.text(
            x0 * w + 2,
            y0 * h + 2,
            cls,
            color="white",
            fontsize=7,
            va="top",
            bbox={"fc": color, "ec": "none", "pad": 1},
        )

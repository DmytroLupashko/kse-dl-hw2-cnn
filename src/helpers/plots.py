"""Notebook figures: shared style (dataviz reference palette, light mode), saving, PR curves."""

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from helpers.config import CLASSES
from core.metrics import pr_curve

INK, PAPER, GRID = "#52514e", "#fcfcfb", "#e1e0d9"


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


def save_figures(figures, folder):
    """{file name: figure} saved to `folder`."""

    for fname, fig in figures.items():
        fig.savefig(folder / fname, dpi=120)


def pr_curves(results, colors, classes=CLASSES, title="Out-of-fold rows"):
    """PR curve per class of pooled OOF rows of {name: `CVResult`}, `colors` {name: color}; AP in legend, dot at
    best-F1 threshold."""

    fig, axes = plt.subplots(
        1, len(classes), figsize=(3.4 * len(classes), 3.6), sharey=True, squeeze=False
    )

    for cls, ax in zip(classes, axes[0]):
        for name, r in results.items():
            y, p = r.oof[cls], r.oof[f"p_{cls}"]
            precision, recall, _, f1 = pr_curve(y, p)
            best = f1.argmax()
            color = colors[name]
            label = f"{name}, AP {average_precision_score(y, p):.3f}"
            ax.plot(recall, precision, color=color, lw=2, drawstyle="steps-post", label=label)
            ax.plot(recall[best], precision[best], "o", ms=7, color=color, mec=PAPER, mew=2)

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

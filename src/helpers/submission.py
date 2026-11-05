"""Submission from trained CV checkpoints: thresholds on OOF rows, test probs, thresholded CSV. Decisions: `core.predict`.

`Single(path)`: test probs of model config `path` averaged over folds, best-F1 thresholds. Ensembles: `helpers.ensemble`.
CLI: `python -m helpers.submission <config.yaml> ['--thresholds=[T,T,T,T,T]'] [--submit=true]`
Notebook: `Single(path).write()`.
CSV saved to `path` (`Single` default `data/submissions/<experiment>__<name>.csv`), unthresholded test probs next to it
as `.parquet` (for ensembling).
"""

import subprocess
from pathlib import Path

import pandas as pd
from jsonargparse import auto_cli

from core import predict
from core.dataset import read_test
from .config import CLASSES, SUBMISSIONS
from .cv import CVResult
from .tracking import PROB_COLS


class Submission:
    """CSV `path` of thresholded test probs. Subclass: `thresholds()` per class (from OOF rows) and `test_probs(test)`."""

    def __init__(self, path):
        self.path = Path(path)

    def write(self, test=None):
        """Thresholds, then probs of `test` rows (default test.csv); writes CSV `path`
        (`image_id` plus one 0/1 column per class, rows in `test` order) and probs `.parquet`; returns `path`."""

        thresholds = self.thresholds()
        print(pd.Series(thresholds, index=CLASSES, name="threshold").round(3))

        if test is None:
            test = read_test()

        probs = self.test_probs(test)
        submission = test[["image_id"]].copy()
        submission[CLASSES] = (probs >= thresholds).astype(int)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(self.path, index=False)
        print(f"saved {self.path}: {len(submission):,} rows")

        out = submission[["image_id"]].copy()
        out[PROB_COLS] = probs
        out.to_parquet(self.path.with_suffix(".parquet"))

        return self.path

    def submit(self, message):
        """Uploads written CSV to Kaggle competition."""

        subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "cats-and-dogs",
                "-f",
                str(self.path),
                "-m",
                message,
            ],
            check=True,
        )


class Single(Submission):
    """One model: saved CV run of model config `config`, test probs averaged over fold models, `best_f1` thresholds on
    pooled OOF rows (`fixed_thresholds` given: one value per class in `CLASSES` order instead). Config saved at training
    used: CLI overrides kept."""

    def __init__(self, config, path=None, fixed_thresholds=None):
        self.result = CVResult.load(config)
        self.config = self.result.config
        self.fixed_thresholds = fixed_thresholds
        suffix = (
            ""
            if fixed_thresholds is None
            else "__fixed_" + "_".join(map(str, fixed_thresholds))
        )

        super().__init__(
            path
            or SUBMISSIONS
            / f"{self.config['experiment']}__{self.config['name']}{suffix}.csv"
        )

    def thresholds(self):
        return predict.single_thresholds(
            self.result.labels, self.result.probs, self.fixed_thresholds
        )

    def test_probs(self, test):
        """Mean sigmoid probs over fold models, one row per row of `test`."""

        models = (self.result.fold_model(fold) for fold in self.result.folds)

        return predict.test_probs(models, test, self.config["data"])


def main(config: str, thresholds: list[float] | None = None, submit: bool = False):
    """CSV of model config `config`; `thresholds`: fixed one per class (default OOF best-F1); `submit`: upload to Kaggle."""

    if thresholds is not None and len(thresholds) != len(CLASSES):
        raise SystemExit(
            f"--thresholds: {len(CLASSES)} values, one per class in {CLASSES}"
        )

    submission = Single(config, fixed_thresholds=thresholds)
    submission.write()

    if submit:
        rule = (
            "OOF best-F1 thresholds"
            if thresholds is None
            else f"fixed thresholds {thresholds}"
        )
        result = submission.result
        submission.submit(
            f"{result.config['experiment']} / {result.config['name']}, folds {result.folds}, {rule}"
        )


if __name__ == "__main__":
    auto_cli(main)

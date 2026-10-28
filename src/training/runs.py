"""Run folder format: one LightningCLI run (`training.cli`) per fold, read back by `CVResult.load`.

`run_dir(experiment, name, fold)`: `data/processed/cv/<experiment>/<name>/fold_<k>/` (`full/`: every row, no validation).
Holds checkpoint, `config.yaml` (resolved config), `oof.parquet` (validation rows: fold, labels, `p_<class>` probs;
none in `full/`), `run.json` (MLflow run id, written last: run finished).
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from .config import CLASSES, PROCESSED

PROB_COLS = [f"p_{c}" for c in CLASSES]


def read_config(path):
    """Model config `path` as written: top `experiment`, `name`, sections without defaults filled in."""
    return yaml.safe_load(Path(path).read_text())


def run_dir(experiment, name, fold):
    """Run folder of `fold` (None: run on every row)."""
    return (
        PROCESSED
        / "cv"
        / experiment
        / name
        / ("full" if fold is None else f"fold_{fold}")
    )


def save_run(out, run_id, oof=None):
    """Outputs of finished run in run folder `out`: `oof` rows (validation fold only), then `run.json`."""
    if oof is not None:
        oof.to_parquet(out / "oof.parquet")

    (out / "run.json").write_text(json.dumps({"run_id": run_id}))


def load_runs(path):
    """Finished folds of model config `path` (its `experiment`, `name` locate them): resolved config (first fold),
    OOF rows of every fold, {fold: MLflow run id}, {fold: run folder}."""
    top = read_config(path)
    out = run_dir(top["experiment"], top["name"], 0).parent
    dirs = {
        int(d.name.removeprefix("fold_")): d
        for d in out.glob("fold_*")
        if (d / "run.json").exists()
    }

    if not dirs:
        raise FileNotFoundError(f"no finished fold in {out}")

    dirs = dict(sorted(dirs.items()))

    return (
        read_config(next(iter(dirs.values())) / "config.yaml"),
        pd.concat([pd.read_parquet(d / "oof.parquet") for d in dirs.values()]),
        {
            fold: json.loads((d / "run.json").read_text())["run_id"]
            for fold, d in dirs.items()
        },
        dirs,
    )

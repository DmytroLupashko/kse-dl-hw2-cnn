"""Run folder format and MLflow tracking, one database for every notebook and training run.

Run folder: one LightningCLI run (`core.cli`) per fold, read back by `CVResult.load`.
`run_dir(experiment, name, fold)`: `data/processed/cv/<experiment>/<name>/fold_<k>/` (`full/`: every row, no validation).
Holds checkpoint, `config.yaml` (resolved config), `oof.parquet` (validation rows: fold, labels, `p_<class>` probs;
none in `full/`), `run.json` (MLflow run id, written last: run finished).

Fold runs: `run_logger` (run folder, Lightning `MLFlowLogger` at `MLFLOW_URI`), `SaveConfig` (`config.yaml`); used by `core.cli`.
`start_run(experiment, run_name)`: notebook runs next to fold runs.
`curves(run_ids, keys)`: metrics of every finished fold run (`CVResult.mean_curve`, `CVResult.history`).
Inside notebook run: plain `mlflow.log_*`.
"""

import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger

from .config import CLASSES, PROCESSED, ROOT

PROB_COLS = [f"p_{c}" for c in CLASSES]

MLFLOW_DIR = ROOT / "data" / "mlflow"  # database and artifacts
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"  # one tracking database for every notebook


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


def artifact_location(experiment):
    """Artifact folder of new `experiment` under `MLFLOW_DIR`, not `./mlruns` of current folder."""

    return (MLFLOW_DIR / "artifacts" / experiment).as_uri()


def run_logger(experiment, name, fold, subcommand):
    """Run folder of `fold` (`run_dir`, created; `fit` empties it first) and its MLflow logger: (root, logger)."""

    root = run_dir(experiment, name, fold)

    if subcommand == "fit":
        shutil.rmtree(
            root, ignore_errors=True
        )  # old checkpoint would mix with new one

    root.mkdir(parents=True, exist_ok=True)

    logger = MLFlowLogger(
        experiment_name=experiment,
        tracking_uri=MLFLOW_URI,
        artifact_location=artifact_location(experiment),
    )

    return root, logger


class SaveConfig(SaveConfigCallback):
    """Resolved config as `config.yaml` in run folder: MLflow logger has no local log dir."""

    def setup(self, trainer, pl_module, stage):
        self.parser.save(
            self.config, Path(trainer.default_root_dir) / "config.yaml", overwrite=True
        )


def start_run(experiment, run_name, params=None):
    """Active notebook run `run_name` of `experiment` (created if missing), `params` logged; `mlflow.ActiveRun`:
    `with` block or `mlflow.end_run()` ends it."""

    mlflow.set_tracking_uri(MLFLOW_URI)

    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_location(experiment))

    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=run_name)

    if params:
        mlflow.log_params(params)

    return run


def curves(run_ids, keys):
    """Per-epoch values of metrics `keys` of runs {fold: run id}, long format (fold, epoch, key, value);
    epoch-level metrics: one value per epoch, in step order."""

    client = mlflow.MlflowClient(MLFLOW_URI)
    rows = []

    for fold, run_id in run_ids.items():
        for key in keys:
            history = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
            rows += [
                {"fold": fold, "epoch": i + 1, "key": key, "value": m.value}
                for i, m in enumerate(history)
            ]

    return pd.DataFrame(rows, columns=["fold", "epoch", "key", "value"])

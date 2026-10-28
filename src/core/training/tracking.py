"""MLflow tracking, one database for every notebook and training run.

Fold runs: Lightning `MLFlowLogger` at `MLFLOW_URI` (`core.training.cli`).
`start_run(experiment, run_name)`: notebook runs next to fold runs.
`TrackedRun(run_id)`: metrics read from finished run; `curves(run_ids, keys)`: of every fold run (`CVResult.mean_curve`,
`CVResult.history`).
Inside notebook run: plain `mlflow.log_*`.
"""

import mlflow
import pandas as pd

from .config import ROOT

MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"  # one tracking database for every notebook


def start_run(experiment, run_name, params=None):
    """Active notebook run `run_name` of `experiment` (created if missing), `params` logged; `mlflow.ActiveRun`:
    `with` block or `mlflow.end_run()` ends it."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=run_name)

    if params:
        mlflow.log_params(params)

    return run


class TrackedRun:
    """Existing run `run_id`, e.g. fold run after training."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.client = mlflow.MlflowClient(MLFLOW_URI)

    def curve(self, key):
        """Values of metric `key` in step order; epoch-level metrics: one per epoch."""
        history = self.client.get_metric_history(self.run_id, key)

        return [m.value for m in sorted(history, key=lambda m: m.step)]


def curves(run_ids, keys):
    """Per-epoch values of metrics `keys` of runs {fold: run id}, long format (fold, epoch, key, value)."""
    return pd.DataFrame(
        [
            {"fold": fold, "epoch": i + 1, "key": key, "value": v}
            for fold, run_id in run_ids.items()
            for key in keys
            for i, v in enumerate(TrackedRun(run_id).curve(key))
        ],
        columns=["fold", "epoch", "key", "value"],
    )

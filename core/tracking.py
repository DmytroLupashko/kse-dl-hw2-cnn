"""MLflow tracking, one database for every notebook and training run.

`Experiment(name)`: Lightning logger of fold runs (`core.train`), notebook runs (`NotebookRun`) next to them.
`TrackedRun(run_id)`: metrics written into or read from finished run (`core.cv`, `CVResult.mean_curve`).
Inside notebook run: `run.log_figures`, plain `mlflow.log_*` for the rest.
"""
import mlflow
from lightning.pytorch.loggers import MLFlowLogger

from .config import ROOT

MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"  # one tracking database for every notebook


class Experiment:
    """MLflow experiment `name`, created on first run."""

    def __init__(self, name):
        self.name = name

    def fold_logger(self, run_name):
        """Lightning logger: new run `run_name`; run id at `.run_id`."""
        return MLFlowLogger(experiment_name=self.name, run_name=run_name, tracking_uri=MLFLOW_URI)

    def run(self, run_name, params=None):
        """Active notebook run `run_name`, `params` logged; `with` block or `.end()` ends it."""
        return NotebookRun(self.name, run_name, params)


class NotebookRun:
    """Active MLflow run started at creation; target of plain `mlflow.log_*` calls until ended."""

    def __init__(self, experiment, run_name, params=None):
        mlflow.set_tracking_uri(MLFLOW_URI)
        self.run = mlflow.start_run(experiment_id=mlflow.set_experiment(experiment).experiment_id, run_name=run_name)
        if params:
            mlflow.log_params(params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.end("FAILED" if exc_type else "FINISHED")  # as `with mlflow.start_run()`

    def end(self, status="FINISHED"):
        mlflow.end_run(status)

    def log_figures(self, figures, folder):
        """{file name: figure} saved to `folder` and logged to this run."""
        for fname, fig in figures.items():
            fig.savefig(folder / fname, dpi=120)
            mlflow.log_figure(fig, fname)


class TrackedRun:
    """Existing run `run_id`, e.g. fold run after training."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.client = mlflow.MlflowClient(MLFLOW_URI)

    def log_metrics(self, metrics):
        """{name: value} into run."""
        for key, value in metrics.items():
            self.client.log_metric(self.run_id, key, value)

    def curve(self, key):
        """Values of metric `key` in step order; epoch-level metrics: one per epoch."""
        history = self.client.get_metric_history(self.run_id, key)
        return [m.value for m in sorted(history, key=lambda m: m.step)]

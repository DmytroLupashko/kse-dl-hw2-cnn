"""LightningCLI of one training run: one fold of model config, or every row.

`python -m core.cli fit --config configs/expNN_<slug>/<model>.yaml --data.fold=<k>`; every row, no
validation: `--data.fold=null`. Any field overridable: `--trainer.max_epochs=1`, `--model.lr=3e-4`.
Config: top `experiment` (MLflow experiment), `name` (model label), sections `model` (`LitCNNMultiLabel`), `data`
(`KaggleDataModule`), `trainer`. Outputs in run folder (`helpers.tracking`), read by `CVResult.load`.
Notebook previews of training draws: `data_module(path, fold)`.
"""

import logging
import warnings

# before imports: tqdm warns at import time
warnings.filterwarnings("ignore", message="IProgress not found")  # no ipywidgets
warnings.filterwarnings(
    "ignore", message=".*does not have many workers"
)  # num_workers = 0 on purpose
warnings.filterwarnings(
    "ignore", message=".*LeafSpec.*is deprecated"
)  # inside lightning
warnings.filterwarnings(
    "ignore", message=".*exists and is not empty"
)  # run folder holds `config.yaml` first

from huggingface_hub.utils import logging as hf_logging
from jsonargparse import ArgumentParser
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import LightningCLI

from core.dataset import KaggleDataModule
from helpers.config import SEED
from core.model import LitCNNMultiLabel, RunCheckpoint
from helpers.progress import EpochBars
from helpers.tracking import SaveConfig, read_config, run_logger

hf_logging.set_verbosity_error()  # HF_TOKEN notice when timm downloads weights


def data_module(path, fold):
    """`KaggleDataModule` of `data` section of model config `path`, validation fold `fold`."""

    parser = ArgumentParser()
    parser.add_class_arguments(KaggleDataModule, "data")
    config = parser.parse_object(
        {"data": {**read_config(path).get("data", {}), "fold": fold}}
    )

    return parser.instantiate(config).data


class CLI(LightningCLI):
    """Run folder and MLflow logger from `experiment`, `name`, `data.fold` (`helpers.tracking.run_logger`); `fit`
    empties run folder first; no fold: no validation. `model.pos_weight` from `data.pos_weight`."""

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--experiment", required=True, help="MLflow experiment `cnn-expNN-<slug>`"
        )
        parser.add_argument(
            "--name", required=True, help="model label: results folder, plot legends"
        )
        parser.link_arguments("data.pos_weight", "model.pos_weight", apply_on="instantiate")

    def instantiate_trainer(self, **kwargs):
        config = self.config[self.subcommand]
        root, logger = run_logger(
            config.experiment, config.name, config.data.fold, self.subcommand
        )

        if config.data.fold is None:
            kwargs["limit_val_batches"] = 0

        return super().instantiate_trainer(
            default_root_dir=root, logger=logger, **kwargs
        )


if __name__ == "__main__":
    logging.getLogger("lightning.pytorch").setLevel(
        logging.WARNING
    )  # no device, tip, stop info lines
    CLI(
        LitCNNMultiLabel,
        KaggleDataModule,
        seed_everything_default=SEED,
        save_config_callback=SaveConfig,
        auto_configure_optimizers=False,  # `LitCNNMultiLabel.configure_optimizers`
        trainer_defaults={
            "max_epochs": 3,
            "log_every_n_steps": 10,
            "enable_model_summary": False,
            "callbacks": [
                RunCheckpoint(),
                LearningRateMonitor(logging_interval="step"),
                EpochBars(),
            ],
        },
    )

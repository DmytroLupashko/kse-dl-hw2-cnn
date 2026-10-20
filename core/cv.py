"""Cross-validation run of one config: training per fold, OOF probs, per-fold AP, saved to disk.

One TOML config per model: `sections/<section>/configs/expNN_<slug>/<model>.toml`.
CLI: `python -m core.cv <config.toml> --epochs 1 --folds 0`
Notebook: `CrossValidator(Config.from_toml(path)).run()` trains; `CVResult.load(config)` only reads saved results.
Results: `data/processed/cv/<experiment>/<name>/`.
"""
import argparse
import dataclasses
import gc
import json
import tomllib
import types
import warnings
from pathlib import Path

# before imports: tqdm warns at import time
warnings.filterwarnings("ignore", message="IProgress not found")  # no ipywidgets
warnings.filterwarnings("ignore", message=".*does not have many workers")  # num_workers = 0 on purpose
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated")  # inside lightning

import mlflow
import pandas as pd
import timm
import torch
from huggingface_hub.utils import logging as hf_logging

from core.config import CHECKPOINTS, CLASSES, MLFLOW_URI, PROCESSED, SEED, VIP
from core.data import TRAIN_TFMS, KaggleDataModule, read_train, sample_weights, val_tfms
from core.model import TORCHVISION_MODELS, LitCNNMultiLabel, pos_weight
from core.predict import ap_report, predict_probs
from core.train import train
from core.wsdan import WSDAN, LitWSDAN

PROB_COLS = [f"p_{c}" for c in CLASSES]
hf_logging.set_verbosity_error()  # HF_TOKEN notice when timm downloads weights
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclasses.dataclass(frozen=True)
class Config:
    name: str  # model label: run names, plot legends
    experiment: str  # `cnn-<section>-<slug>`
    timm_model: str | None = None  # exactly one of `timm_model`, `torchvision_model`
    torchvision_model: str | None = None  # key of `core.model.TORCHVISION_MODELS`, for models absent in timm
    wsdan: bool = False  # WS-DAN on `timm_model` (`core.wsdan`): attention head, attention crop and drop
    folds: list[int] = dataclasses.field(default_factory=lambda: [0, 1])
    epochs: int = 3
    img_size: int = 64
    augment: str = "default"  # key of `core.data.TRAIN_TFMS`
    oversample: int = 1  # VIP rows drawn this many times per epoch, VIP `pos_weight` divided by it; 1: plain shuffling
    downsample: float = 1.0  # non-VIP rows drawn this fraction of times per epoch, `pos_weight` counted over draws
    pos_weight_power: float = 0.75  # `pos_weight` = ((N - n_c) / n_c) ** power; 1.0: full ratio
    n_other: int | None = None  # non-VIP rows kept per fold; None: full data
    batch_size: int = 32
    num_workers: int = 0
    initial_lr: float = 1e-3
    final_lr: float = 1e-5
    retrain: bool = False  # False: finished fold runs restored from checkpoints, unfinished ones trained
    log_every_n_steps: int = 10

    def __post_init__(self):
        if (self.timm_model is None) == (self.torchvision_model is None):
            raise ValueError(f"{self.name}: set exactly one of timm_model, torchvision_model")
        if self.wsdan and self.timm_model is None:
            raise ValueError(f"{self.name}: wsdan needs timm_model")

    @classmethod
    def from_toml(cls, path, **overrides):
        return cls(**{**tomllib.loads(Path(path).read_text()), **overrides})

    @classmethod
    def from_cli(cls, argv=None):
        """Config file plus one optional `--<field>` override per field."""
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("config", type=Path)
        for field in dataclasses.fields(cls):
            flag = "--" + field.name.replace("_", "-")
            if field.type is bool:
                parser.add_argument(flag, dest=field.name, action=argparse.BooleanOptionalAction, default=None)
            elif field.type == list[int]:
                parser.add_argument(flag, dest=field.name, type=int, nargs="+")
            else:
                kind = field.type
                if isinstance(kind, types.UnionType):  # `int | None`
                    kind = next(t for t in kind.__args__ if t is not type(None))
                parser.add_argument(flag, dest=field.name, type=kind)
        args = vars(parser.parse_args(argv))
        path = args.pop("config")
        return cls.from_toml(path, **{k: v for k, v in args.items() if v is not None})

    @property
    def results_dir(self):
        return PROCESSED / "cv" / self.experiment / self.name

    def read_data(self):
        """Train table; `n_other` set: every VIP row plus `n_other` others per fold, so every class has positives in every fold."""
        df = read_train()
        if self.n_other is None:
            return df
        vip = df[VIP].any(axis=1)
        other = df[~vip].groupby("fold").sample(self.n_other, random_state=SEED)
        return pd.concat([df[vip], other])  # index kept: OOF rows map back to `read_train()` rows

    def fold_model(self, fold):
        """Trained `LitCNNMultiLabel` of `fold` from its checkpoint, on `DEVICE`."""
        ckpt = next((CHECKPOINTS / self.experiment / f"{self.name} fold {fold}").glob("*.ckpt"))
        # pos_weight buffer restored from checkpoint; passed only for its shape
        kwargs = self.lit_kwargs(torch.ones(len(CLASSES)))
        return self.lit_class.load_from_checkpoint(ckpt, map_location=DEVICE, **kwargs)

    def predict_val(self, fold, tfms):
        """(probs, labels) of validation rows of `fold` by its trained network, images through `tfms` instead of `val_tfms`."""
        dm = KaggleDataModule(self.read_data(), fold=fold, batch_size=self.batch_size, num_workers=self.num_workers, val_tfms=tfms)
        dm.setup()
        return predict_probs(self.fold_model(fold), dm.val_dataloader())

    @property
    def lit_class(self):
        return LitWSDAN if self.wsdan else LitCNNMultiLabel

    def lit_kwargs(self, weights):
        if self.wsdan:
            model_class = WSDAN
            model_kwargs = {"model_name": self.timm_model, "num_classes": len(CLASSES)}
        elif self.timm_model is not None:
            model_class = timm.create_model
            model_kwargs = {"model_name": self.timm_model, "pretrained": True, "num_classes": len(CLASSES)}
        else:
            model_class = TORCHVISION_MODELS[self.torchvision_model]
            model_kwargs = {"num_classes": len(CLASSES)}
        return {
            "model_class": model_class,
            "model_kwargs": model_kwargs,
            "initial_lr": self.initial_lr,
            "final_lr": self.final_lr,
            "pos_weight": weights,
        }


@dataclasses.dataclass
class CVResult:
    config: Config
    oof: pd.DataFrame  # validation rows of `config.folds`: fold, labels, `p_<class>` probs
    fold_ap: pd.DataFrame  # AP per class plus mAP, one row per fold
    run_ids: dict[int, str]  # fold: MLflow run id

    @property
    def labels(self):
        return self.oof[CLASSES].to_numpy()

    @property
    def probs(self):
        return self.oof[PROB_COLS].to_numpy()

    def save(self):
        out = self.config.results_dir
        out.mkdir(parents=True, exist_ok=True)
        self.oof.to_parquet(out / "oof.parquet")
        self.fold_ap.to_parquet(out / "fold_ap.parquet")
        meta = {"config": dataclasses.asdict(self.config), "run_ids": self.run_ids}
        (out / "cv.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, config):
        """Saved result of `config`; only `experiment` and `name` locate it.
        Stored config returned, so CLI overrides used at training kept."""
        out = config.results_dir
        meta = json.loads((out / "cv.json").read_text())
        return cls(
            Config(**meta["config"]),
            pd.read_parquet(out / "oof.parquet"),
            pd.read_parquet(out / "fold_ap.parquet"),
            {int(fold): run_id for fold, run_id in meta["run_ids"].items()},
        )


class CrossValidator:
    """Trains every fold of `config.folds` on `config.read_data()`."""

    def __init__(self, config):
        self.config = config
        self.df = config.read_data()

    def run(self):
        """Trains all folds, saves result to `config.results_dir`, returns it."""
        oof, fold_ap, run_ids = [], {}, {}
        for fold in self.config.folds:
            val, fold_ap[fold], run_ids[fold] = self.fit_fold(fold)
            oof.append(val)
        result = CVResult(self.config, pd.concat(oof), pd.DataFrame(fold_ap).T.rename_axis("fold"), run_ids)
        result.save()
        return result

    def fit_fold(self, fold):
        """Returns (validation rows with probs, AP report, MLflow run id) of `fold`."""
        cfg = self.config
        is_val = (self.df["fold"] == fold).to_numpy()
        train_part = self.df.loc[~is_val]
        if cfg.downsample == 1:
            weights = pos_weight(train_part[CLASSES].to_numpy(), cfg.pos_weight_power)  # train part of fold only
            weights[[CLASSES.index(c) for c in VIP]] /= cfg.oversample  # sampling x pos_weight kept as without oversampling
        else:  # positives and negatives counted as sampler draws them
            draws = sample_weights(train_part, cfg.oversample, cfg.downsample)
            weights = pos_weight(train_part[CLASSES].to_numpy(), cfg.pos_weight_power, draws)
        dm = KaggleDataModule(
            self.df,
            fold=fold,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            train_tfms=TRAIN_TFMS[cfg.augment](cfg.img_size),
            val_tfms=val_tfms(cfg.img_size),
            oversample=cfg.oversample,
            downsample=cfg.downsample,
        )
        best_path, run_id = train(
            cfg.lit_class(**cfg.lit_kwargs(weights)),
            dm,
            experiment=cfg.experiment,
            run_name=f"{cfg.name} fold {fold}",
            epochs=cfg.epochs,
            reuse=not cfg.retrain,
            log_every_n_steps=cfg.log_every_n_steps,
        )
        best = cfg.lit_class.load_from_checkpoint(best_path, map_location=DEVICE, **cfg.lit_kwargs(weights))
        dm.setup()  # setup runs inside fit only, skipped when run reused
        probs, y = predict_probs(best, dm.val_dataloader())
        del best
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()
        ap = ap_report(y, probs)
        client = mlflow.MlflowClient(MLFLOW_URI)
        for c, v in ap.items():
            client.log_metric(run_id, f"best_ap_{c}", v)
        val = self.df.loc[is_val, ["fold", *CLASSES]].copy()
        val[PROB_COLS] = probs
        return val, ap, run_id


if __name__ == "__main__":
    result = CrossValidator(Config.from_cli()).run()
    print(result.fold_ap.round(3))
    print(f"saved: {result.config.results_dir}")

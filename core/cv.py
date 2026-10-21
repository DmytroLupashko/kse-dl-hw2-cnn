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

import numpy as np
import pandas as pd
import torch
from huggingface_hub.utils import logging as hf_logging

from core.augment import RECIPES, Validation
from core.config import CLASSES, DEVICE, PROCESSED, SEED, VIP
from core.data import KaggleDataModule, read_train
from core.model import LitCNNMultiLabel
from core.scores import Scores
from core.tracking import TrackedRun
from core.train import TrainingRun

PROB_COLS = [f"p_{c}" for c in CLASSES]
hf_logging.set_verbosity_error()  # HF_TOKEN notice when timm downloads weights


@dataclasses.dataclass(frozen=True)
class Config:
    name: str  # model label: run names, plot legends
    experiment: str  # `cnn-<section>-<slug>`
    timm_model: str | None = None  # exactly one of `timm_model`, `torchvision_model`
    torchvision_model: str | None = None  # key of `core.model.TORCHVISION_MODELS`, for models absent in timm
    folds: list[int] = dataclasses.field(default_factory=lambda: [0, 1])
    epochs: int = 3
    img_size: int = 64
    augment: str = "default"  # key of `core.augment.RECIPES`
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

    def run(self, fold):
        """`TrainingRun` of `fold`: MLflow run and checkpoint folder `<name> fold <fold>`."""
        return TrainingRun(self.experiment, f"{self.name} fold {fold}")

    def fold_model(self, fold):
        """Trained `LitCNNMultiLabel` of `fold` from its checkpoint, on `DEVICE`."""
        ckpt = self.run(fold).best_checkpoint()
        return LitCNNMultiLabel.load(ckpt, self.timm_model, self.torchvision_model, map_location=DEVICE)

    def predict_val(self, fold, tfms):
        """(probs, labels) of validation rows of `fold` by its trained network, images through `tfms` instead of `Validation`."""
        dm = KaggleDataModule(self.read_data(), fold=fold, batch_size=self.batch_size, num_workers=self.num_workers, val_tfms=tfms)
        dm.setup()
        return self.fold_model(fold).predict_probs(dm.val_dataloader())


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

    @property
    def scores(self):
        """`Scores` of pooled OOF rows."""
        return Scores(self.labels, self.probs)

    def mean_curve(self, key):
        """Per-epoch metric `key` of fold runs, averaged over folds."""
        return np.mean([TrackedRun(run_id).curve(key) for run_id in self.run_ids.values()], axis=0)

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
        fields = {f.name for f in dataclasses.fields(Config)}  # fields removed since saving dropped, e.g. `wsdan`
        return cls(
            Config(**{k: v for k, v in meta["config"].items() if k in fields}),
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
        dm = KaggleDataModule(
            self.df,
            fold=fold,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            train_tfms=RECIPES[cfg.augment](cfg.img_size).compose(),
            val_tfms=Validation(cfg.img_size).compose(),
            oversample=cfg.oversample,
            downsample=cfg.downsample,
        )
        model = LitCNNMultiLabel.build(
            cfg.timm_model, cfg.torchvision_model, cfg.initial_lr, cfg.final_lr, dm.pos_weight(cfg.pos_weight_power)
        )
        run_id = cfg.run(fold).fit(
            model, dm, cfg.epochs, reuse=not cfg.retrain, log_every_n_steps=cfg.log_every_n_steps
        )
        del model
        best = cfg.fold_model(fold)
        dm.setup()  # setup runs inside fit only, skipped when run reused
        probs, y = best.predict_probs(dm.val_dataloader())
        del best
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()
        ap = Scores(y, probs).ap()
        TrackedRun(run_id).log_metrics({f"best_ap_{c}": v for c, v in ap.items()})
        val = self.df.loc[is_val, ["fold", *CLASSES]].copy()
        val[PROB_COLS] = probs
        return val, ap, run_id


if __name__ == "__main__":
    result = CrossValidator(Config.from_cli()).run()
    print(result.fold_ap.round(3))
    print(f"saved: {result.config.results_dir}")

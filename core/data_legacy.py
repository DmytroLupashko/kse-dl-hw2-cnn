"""Legacy sampling and data module: finished runs up to exp40 trained with these (`Config.is_new = False`). Frozen, do not change."""

import lightning.pytorch as pl
import numpy as np
import torch

from .augment import Default, Validation
from .config import CLASSES, TRAIN_DIR, VIP
from .data import MultilabelImageDataset


class LegacySampling:
    """Training rows per epoch and BCE `pos_weight` matched to them: ((N - n_c) / n_c) ** `power`.
    Base counts N, n_c over sampler draws and samples by `sample_weights`; subclasses change only what differs.
    """

    power = 0.75  # damped ratio without oversampling

    def __init__(self, oversample, downsample):
        self.oversample = oversample
        self.downsample = downsample

    def sample_weights(self, df):
        """Sampler weight of each row of `df`, equal to its expected draws per epoch: `oversample` for rows with any VIP label, `downsample` for others."""
        return np.where(df[VIP].any(axis=1), self.oversample, self.downsample)

    def pos_weight(self, df):
        counts = self.sample_weights(df)
        n_pos = counts @ df[CLASSES].to_numpy()
        return torch.tensor(
            ((counts.sum() - n_pos) / n_pos) ** self.power, dtype=torch.float32
        )

    def loader(self, ds, batch_size, num_workers):
        # augmentation runs per draw in `__getitem__`, so every copy is new view

        weights = self.sample_weights(ds.df)
        num_samples = round(weights.sum())

        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True
        )

        return ds.loader(batch_size, num_workers, sampler=sampler)


class LegacyShuffle(LegacySampling):
    """Every row once per epoch, plain shuffling."""

    def loader(self, ds, batch_size, num_workers):
        return ds.loader(batch_size, num_workers, shuffle=True)


class LegacyOversample(LegacySampling):
    """VIP rows `oversample` times per epoch. N, n_c counted over rows, VIP weight divided by `oversample`:
    sampling x weight kept as without oversampling."""

    power = 1.0

    def pos_weight(self, df):
        n_pos = df[CLASSES].to_numpy().sum(axis=0)
        weights = torch.tensor(
            ((len(df) - n_pos) / n_pos) ** self.power, dtype=torch.float32
        )
        weights[[CLASSES.index(c) for c in VIP]] /= self.oversample
        return weights


class LegacyDownsample(LegacySampling):
    """Non-VIP rows `downsample` fraction of times per epoch, new random subset each epoch."""


class LegacyOverDownsample(LegacySampling):
    """VIP rows `oversample` times, non-VIP rows `downsample` fraction of times per epoch."""

    power = 1.0


LEGACY_SAMPLINGS = {  # (oversample != 1, downsample != 1)
    (False, False): LegacyShuffle,
    (True, False): LegacyOversample,
    (False, True): LegacyDownsample,
    (True, True): LegacyOverDownsample,
}


class LegacyKaggleDataModule(pl.LightningDataModule):
    """Fold `fold` of `df` as validation, the rest as training. `fold=None`: train on all rows, no validation.
    `oversample` > 1: training rows with any VIP label drawn that many times per epoch (with replacement), others once.
    `downsample` < 1: other rows drawn that fraction of times per epoch, new random subset each epoch.
    Sampling and `pos_weight` delegated to `LEGACY_SAMPLINGS` class picked by these two."""

    is_new = False  # `TrainingRun.recipe` key absent: hashes of finished runs unchanged

    def __init__(
        self,
        df,
        fold=0,
        batch_size=32,
        num_workers=4,
        train_tfms=Default().compose(),
        val_tfms=Validation().compose(),
        oversample=1,
        downsample=1.0,
    ):
        super().__init__()
        self.df = df
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms
        self.oversample = oversample
        self.downsample = downsample
        self.sampling = LEGACY_SAMPLINGS[oversample != 1, downsample != 1](
            oversample, downsample
        )

    def setup(self, stage=None):
        is_val = self.df["fold"] == self.fold
        self.train_ds = MultilabelImageDataset(
            self.df[~is_val], TRAIN_DIR, self.train_tfms
        )
        self.val_ds = MultilabelImageDataset(self.df[is_val], TRAIN_DIR, self.val_tfms)

    def pos_weight(self):
        """BCE `pos_weight` per class from training rows of fold."""
        return self.sampling.pos_weight(self.df[self.df["fold"] != self.fold])

    def train_dataloader(self):
        return self.sampling.loader(self.train_ds, self.batch_size, self.num_workers)

    def val_dataloader(self):
        if self.fold is None:
            return []
        return self.val_ds.loader(self.batch_size, self.num_workers)

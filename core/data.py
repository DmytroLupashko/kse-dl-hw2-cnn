"""Train/test tables, image dataset and Lightning data module; transforms in `core.augment`."""

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image

from .augment import Default, RowTransform, Validation
from .config import CLASSES, DATA, N_FOLDS, SEED, TRAIN_DIR, VIP


def read_train():
    """train.csv plus `fold` column: multilabel stratified K-fold."""
    df = pd.read_csv(DATA / "train.csv")
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(mskf.split(df, df[CLASSES].values)):
        df.loc[val_idx, "fold"] = fold
    return df


def read_test():
    return pd.read_csv(DATA / "test.csv")


class MultilabelImageDataset(torch.utils.data.Dataset):
    """(image, labels) per row of `df`; test rows have no labels, so zeros."""

    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = RowTransform(transform, self.df, img_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(
            Image.open(self.img_dir / f"{row['image_id']}.jpg").convert("RGB")
        )
        image = self.transform(image, row)
        has_labels = set(CLASSES) <= set(row.index)
        labels = (
            row[CLASSES].to_numpy(np.float32)
            if has_labels
            else np.zeros(len(CLASSES), np.float32)
        )
        return image, torch.from_numpy(labels)

    def loader(self, batch_size, num_workers, shuffle=False, sampler=None):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )


class Sampling:
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


class Shuffle(Sampling):
    """Every row once per epoch, plain shuffling."""

    def loader(self, ds, batch_size, num_workers):
        return ds.loader(batch_size, num_workers, shuffle=True)


class Oversample(Sampling):
    """VIP rows `oversample` times per epoch. N, n_c counted over rows, VIP weight divided by `oversample`:
    sampling x weight kept as without oversampling."""

    power = 0.5


class Downsample(Sampling):
    """Non-VIP rows `downsample` fraction of times per epoch, new random subset each epoch."""


class OverDownsample(Sampling):
    """VIP rows `oversample` times, non-VIP rows `downsample` fraction of times per epoch."""

    power = 1.0


SAMPLINGS = {  # (oversample != 1, downsample != 1)
    (False, False): Shuffle,
    (True, False): Oversample,
    (False, True): Downsample,
    (True, True): OverDownsample,
}


class KaggleDataModule(pl.LightningDataModule):
    """Fold `fold` of `df` as validation, the rest as training. `fold=None`: train on all rows, no validation.
    `oversample` > 1: training rows with any VIP label drawn that many times per epoch (with replacement), others once.
    `downsample` < 1: other rows drawn that fraction of times per epoch, new random subset each epoch.
    Sampling and `pos_weight` delegated to `SAMPLINGS` class picked by these two."""

    is_new = True  # `TrainingRun.recipe` adds class name, so runs never reuse legacy checkpoints

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
        self.sampling = SAMPLINGS[oversample != 1, downsample != 1](
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

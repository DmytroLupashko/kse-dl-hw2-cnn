"""Train/test tables, image dataset and Lightning data module; transforms in `dataset.augment`."""

from typing import Callable

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image

from training.config import CLASSES, DATA, N_FOLDS, SEED, TRAIN_DIR, VIP
from .augment import Default, Recipe, Validation

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


def sample_weights(df, oversample, downsample):
    """Expected draws per epoch of each row of `df`: `oversample` for rows with any VIP label, `downsample` for others."""
    return np.where(df[VIP].any(axis=1), oversample, downsample)


class MultilabelImageDataset(torch.utils.data.Dataset):
    """(image, labels) per row of `df`; test rows have no labels, so zeros."""

    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(
            Image.open(self.img_dir / f"{row['image_id']}.jpg").convert("RGB")
        )
        has_labels = set(CLASSES) <= set(row.index)
        # read by `NonVipDropout`, ignored by other transforms
        vip = has_labels and bool(row[VIP].any())
        image = self.transform(image=image, vip=vip)["image"]
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


def sampled_loader(ds, batch_size, num_workers, oversample, downsample):
    """Training loader of `ds`: plain shuffling when both 1, else rows drawn by `sample_weights` with replacement."""
    if oversample == 1 and downsample == 1:
        return ds.loader(batch_size, num_workers, shuffle=True)

    # augmentation runs per draw in `__getitem__`, so every copy is new view
    weights = sample_weights(ds.df, oversample, downsample)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=round(weights.sum()), replacement=True)

    return ds.loader(batch_size, num_workers, sampler=sampler)


class KaggleDataModule(pl.LightningDataModule):
    """Fold `fold` of `read_train()` as validation, the rest as training; `fold` None: every row training, no validation.
    `augment`: recipe class (YAML `class_path`) built with `img_size`; validation: `Validation(img_size)`.
    `oversample` > 1: training rows with any VIP label drawn that many times per epoch (with replacement), others once.
    `downsample` < 1: other rows drawn that fraction of times per epoch, new random subset each epoch."""

    def __init__(
        self,
        fold: int | None,
        augment: Callable[[int], Recipe] = Default,
        img_size: int = 64,
        oversample: int = 1,
        downsample: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["num_workers"])
        self.df = read_train()
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_tfms = augment(img_size).compose()
        self.val_tfms = Validation(img_size).compose()
        self.oversample = oversample
        self.downsample = downsample

    def setup(self, stage=None):
        is_val = self.df["fold"] == self.fold
        self.train_ds = MultilabelImageDataset(
            self.df[~is_val], TRAIN_DIR, self.train_tfms
        )
        self.val_ds = MultilabelImageDataset(self.df[is_val], TRAIN_DIR, self.val_tfms)

    def train_dataloader(self):
        return sampled_loader(self.train_ds, self.batch_size, self.num_workers, self.oversample, self.downsample)

    def val_dataloader(self):
        return self.val_ds.loader(self.batch_size, self.num_workers)

"""Train/test tables, image dataset, transforms and Lightning data module."""

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image

from .config import CLASSES, DATA, IMG_SIZE, MEAN, N_FOLDS, SEED, STD, TRAIN_DIR


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


def train_tfms(size=IMG_SIZE):
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ]
    )


def val_tfms(size=IMG_SIZE):
    return A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])


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
        image = self.transform(image=image)["image"]

        has_labels = set(CLASSES) <= set(row.index)
        labels = (
            row[CLASSES].to_numpy(np.float32)
            if has_labels
            else np.zeros(len(CLASSES), np.float32)
        )
        return image, torch.from_numpy(labels)


def make_loader(ds, batch_size, num_workers, shuffle=False):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


class KaggleDataModule(pl.LightningDataModule):
    """Fold `fold` of `df` as validation, the rest as training. `fold=None`: train on all rows, no validation."""

    def __init__(
        self,
        df,
        fold=0,
        batch_size=32,
        num_workers=4,
        train_tfms=train_tfms(),
        val_tfms=val_tfms(),
    ):
        super().__init__()
        self.df = df
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms

    def setup(self, stage=None):
        is_val = self.df["fold"] == self.fold
        self.train_ds = MultilabelImageDataset(
            self.df[~is_val], TRAIN_DIR, self.train_tfms
        )
        self.val_ds = MultilabelImageDataset(self.df[is_val], TRAIN_DIR, self.val_tfms)

    def train_dataloader(self):
        return make_loader(
            self.train_ds, self.batch_size, self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        if self.fold is None:
            return []
        return make_loader(self.val_ds, self.batch_size, self.num_workers)

"""Train/test tables, image dataset, transforms and Lightning data module."""

import albumentations as A
import cv2
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image

from .config import CLASSES, DATA, IMG_SIZE, MEAN, N_FOLDS, PROCESSED, SEED, STD, TRAIN_DIR, VIP

BOXES = PROCESSED / "boxes.csv"  # Grad-CAM pseudo-boxes of training images, written by `python -m core.boxes`
FACE_BOXES = PROCESSED / "boxes_face.csv"  # Grad-CAM hot core of Cat, Dog (model exp21), `python -m core.boxes --face`
HEAD_BOXES = PROCESSED / "boxes_head.csv"  # face boxes, Cat grown to head and ears (model exp24), `python -m core.boxes --head`


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


def lecture_tfms(size=IMG_SIZE):
    """Training augmentations of Advanced CNN lecture (ResNet-18 transfer learning cell)."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ]
    )


def v2_tfms(size=IMG_SIZE):
    """Augmentations v2 (model exp05): orientation, zoom-in crop, color, one degradation, holes."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0), ratio=(1, 1), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=20, p=0.6),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),  # white balance
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(p=0.2),
            A.OneOf(  # at most one degradation per image
                [
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(std_range=(0.02, 0.06)),
                    A.Downscale(scale_range=(0.5, 0.9)),
                ],
                p=0.4,
            ),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(0.05, 0.2),
                hole_width_range=(0.05, 0.2),
                fill="random_uniform",  # black holes invisible on black Moris
                p=0.5,
            ),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ]
    )


def box_safe(tfms):
    """`tfms` with box-safe crop in front. Needs `bboxes`, `box_labels` (pseudo-boxes, `read_boxes`);
    every box kept whole by crop."""
    return with_boxes([A.BBoxSafeRandomCrop(erosion_rate=0.0, p=1), *tfms.transforms])


class EraseBoxes(A.DualTransform):
    """Attention dropping: each pseudo-box filled independently with `p_box`, labels kept; fill zeros, or with
    `random_fill` one random color per box. Box larger than `max_area` of image skipped: likely whole animal, erasing it
    would leave label without animal. `skip_vip`: rows with VIP label untouched (`vip` flag, `MultilabelImageDataset`)."""

    def __init__(self, p_box=0.5, max_area=0.3, random_fill=False, skip_vip=False, p=1.0):
        super().__init__(p=p)
        self.p_box = p_box
        self.max_area = max_area
        self.random_fill = random_fill
        self.skip_vip = skip_vip

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        holes = []
        if self.skip_vip and data.get("vip"):
            return {"holes": holes}
        for x0, y0, x1, y1 in data["bboxes"][:, :4]:  # normalized
            if (x1 - x0) * (y1 - y0) <= self.max_area and self.py_random.random() < self.p_box:
                color = [self.py_random.randint(0, 255) for _ in range(3)] if self.random_fill else 0
                holes.append((int(x0 * w), int(y0 * h), int(np.ceil(x1 * w)), int(np.ceil(y1 * h)), color))
        return {"holes": holes}

    def apply(self, img, holes, **params):
        img = img.copy()
        for x0, y0, x1, y1, color in holes:
            img[y0:y1, x0:x1] = color
        return img

    def apply_to_bboxes(self, bboxes, **params):
        return bboxes


class PasteBoxes(A.DualTransform):
    """Background paste (CMO, model exp17): with `p`, union of pseudo-boxes plus `margin` cut out and pasted at random
    place onto random background, longer side `scale` of background's shorter side; labels kept.
    `backgrounds`: image paths, set by `MultilabelImageDataset` to its rows without labels (train part only)."""

    def __init__(self, margin=0.1, scale=(0.4, 0.9), p=0.5):
        super().__init__(p=p)
        self.margin = margin
        self.scale = scale
        self.backgrounds = []

    def get_params_dependent_on_data(self, params, data):
        boxes = data["bboxes"][:, :4]
        if not len(boxes) or not self.backgrounds:
            return {"paste": None}
        h, w = params["shape"][:2]
        x0, y0 = boxes[:, :2].min(axis=0) - self.margin
        x1, y1 = boxes[:, 2:].max(axis=0) + self.margin
        crop = (int(max(x0, 0) * w), int(max(y0, 0) * h), int(np.ceil(min(x1, 1) * w)), int(np.ceil(min(y1, 1) * h)))
        side = self.py_random.uniform(*self.scale) * min(h, w)
        k = side / max(crop[2] - crop[0], crop[3] - crop[1])
        size = (max(int((crop[2] - crop[0]) * k), 1), max(int((crop[3] - crop[1]) * k), 1))  # (width, height)
        at = (self.py_random.randint(0, w - size[0]), self.py_random.randint(0, h - size[1]))
        return {"paste": (self.py_random.choice(self.backgrounds), crop, size, at)}

    def apply(self, img, paste, **params):
        if paste is None:
            return img
        path, (x0, y0, x1, y1), (pw, ph), (ax, ay) = paste
        h, w = img.shape[:2]
        out = cv2.resize(np.array(Image.open(path).convert("RGB")), (w, h))
        out[ay : ay + ph, ax : ax + pw] = cv2.resize(img[y0:y1, x0:x1], (pw, ph))
        return out

    def apply_to_bboxes(self, bboxes, **params):
        return bboxes  # not used after paste


def with_boxes(transforms, path=BOXES):
    """Compose of `transforms` that gets pseudo-boxes from `path` (`MultilabelImageDataset` passes them)."""
    tfms = A.Compose(
        transforms, bbox_params=A.BboxParams(format="albumentations", label_fields=["box_labels"], clip=True)
    )
    tfms.boxes_path = path
    return tfms


def attention_drop_tfms(size=IMG_SIZE):
    """Default augmentations (exp04) with pseudo-box erase in front (model exp14)."""
    return with_boxes([EraseBoxes(), *train_tfms(size).transforms])


def paste_tfms(size=IMG_SIZE):
    """Default augmentations (exp04) with background paste in front (model exp17)."""
    return with_boxes([PasteBoxes(), *train_tfms(size).transforms])


def grid_dropout_tfms(size=IMG_SIZE):
    """Grid dropout (model exp18): default recipe plus grid dropout, `Affine` on every draw with zoom-out and shear."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            # each hole own random color: black would mimic Moris fur
            A.GridDropout(ratio=0.2, unit_size_range=(size // 10, size // 4), random_offset=True, fill="random_uniform", p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.Affine(
                scale=(0.7, 1.2),
                rotate=(-20, 20),
                shear=(-10, 10),
                translate_percent=0.1,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ]
    )


class NonVipDropout(A.CoarseDropout):
    """`CoarseDropout` on rows without VIP label only; `vip` flag passed by `MultilabelImageDataset`."""

    def get_params_dependent_on_data(self, params, data):
        if data.get("vip"):
            return {"holes": np.empty((0, 4), np.int32), "seed": 0}
        return super().get_params_dependent_on_data(params, data)


def coarse_dropout_non_vip_tfms(size=IMG_SIZE):
    """Default augmentations (exp04) with colored holes on non-VIP rows after flip (model exp20)."""
    resize, flip, *rest = train_tfms(size).transforms
    # random color per hole: black would mimic Moris fur; VIP rows untouched, coat pattern stays (exp18)
    holes = NonVipDropout(
        num_holes_range=(1, 4), hole_height_range=(0.1, 0.25), hole_width_range=(0.1, 0.25), fill="random_uniform", p=0.5
    )
    return A.Compose([resize, flip, holes, *rest])


class NonVipImageCompression(A.ImageCompression):
    """`ImageCompression` on rows without VIP label only; `vip` flag passed by `MultilabelImageDataset`."""

    def get_params_dependent_on_data(self, params, data):
        return {"vip": bool(data.get("vip"))}

    def apply(self, img, quality, image_type, vip=False, **params):
        return img if vip else super().apply(img, quality, image_type, **params)


def jpeg_tfms(size=IMG_SIZE, compression=A.ImageCompression):
    """exp20 recipe with JPEG quality 10 on half of training draws (model exp28); `compression` class picks rows."""
    *tfms, normalize, to_tensor = coarse_dropout_non_vip_tfms(size).transforms
    # last before Normalize: model sees block artefacts as encoded; fur, whiskers, eyes lost first, outline kept
    jpeg = compression(quality_range=(10, 10), p=0.5)
    return A.Compose([*tfms, jpeg, normalize, to_tensor])


def jpeg_non_vip_tfms(size=IMG_SIZE):
    """`jpeg_tfms` on non-VIP rows only."""
    return jpeg_tfms(size, NonVipImageCompression)

class NonVipToGray(A.ToGray):
    """`ToGray` on rows without VIP label only; `vip` flag passed by `MultilabelImageDataset`."""

    def get_params_dependent_on_data(self, params, data):
        return {"vip": bool(data.get("vip"))}

    def apply(self, img, vip=False, **params):
        return img if vip else super().apply(img, **params)


def gray_non_vip_tfms(size=IMG_SIZE):
    """exp20 recipe with gray on half of non-VIP training draws (model exp30)."""
    *tfms, normalize, to_tensor = coarse_dropout_non_vip_tfms(size).transforms
    # after holes: colored holes turn gray too, no color cue left on gray draws
    return A.Compose([*tfms, NonVipToGray(p=0.5), normalize, to_tensor])


def face_erase_tfms(size=IMG_SIZE, path=FACE_BOXES):
    """exp20 recipe with Cat, Dog boxes from `path` erased on non-VIP rows in front (model exp21: face boxes)."""
    # random color: black would mimic Moris fur
    erase = EraseBoxes(random_fill=True, skip_vip=True)
    return with_boxes([erase, *coarse_dropout_non_vip_tfms(size).transforms], path=path)


def head_erase_tfms(size=IMG_SIZE):
    """exp21 recipe with Cat head boxes instead of face boxes (model exp24)."""
    return face_erase_tfms(size, path=HEAD_BOXES)


def v3_no_crop_tfms(size=IMG_SIZE):
    """Augmentations v3 without crop (model exp07): flip, lighting, affine, image quality."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 150), p=0.3),  # gamma > 1 darkens: Moris weak on dark images
            A.PlanckianJitter(mode="blackbody", p=0.3),  # white balance
            A.Affine(
                scale=(0.7, 1.3),
                rotate=(-15, 15),
                balanced_scale=True,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(40, 95)),
                    A.Downscale(scale_range=(0.6, 0.9)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ],
                p=0.4,  # blur cost VIP 0.21-0.29 AP in exp04 robustness test
            ),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ]
    )


def v3_tfms(size=IMG_SIZE):
    """Augmentations v3 (model exp06): box-safe crop plus `v3_no_crop_tfms`."""
    return box_safe(v3_no_crop_tfms(size))


def box_crop_tfms(size=IMG_SIZE):
    """Default augmentations (exp04) with box-safe crop in front (model exp07)."""
    return box_safe(train_tfms(size))


# `Config.augment`: builder taking image size
TRAIN_TFMS = {
    "default": train_tfms,
    "lecture": lecture_tfms,
    "v2": v2_tfms,
    "v3": v3_tfms,
    "v3_no_crop": v3_no_crop_tfms,
    "box_crop": box_crop_tfms,
    "attention_drop": attention_drop_tfms,
    "paste": paste_tfms,
    "grid_dropout": grid_dropout_tfms,
    "coarse_dropout_non_vip": coarse_dropout_non_vip_tfms,
    "jpeg_non_vip": jpeg_non_vip_tfms,
    "jpeg_all": jpeg_tfms,
    "gray_non_vip": gray_non_vip_tfms,
    "face_erase": face_erase_tfms,
    "head_erase": head_erase_tfms,
}


def val_tfms(size=IMG_SIZE):
    return A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])


def read_boxes(path=BOXES):
    """{image_id: [(class, (x_min, y_min, x_max, y_max)), ...]} from `path`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m core.boxes`")
    df = pd.read_csv(path)
    coords = df[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    boxes = {}
    for image_id, cls, box in zip(df["image_id"], df["cls"], coords):
        boxes.setdefault(image_id, []).append((cls, tuple(box)))
    return boxes


class MultilabelImageDataset(torch.utils.data.Dataset):
    """(image, labels) per row of `df`; test rows have no labels, so zeros."""

    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # transform with `bbox_params` (v3) gets each image's pseudo-boxes
        self.boxes = read_boxes(transform.boxes_path) if "bboxes" in transform.processors else None
        for t in transform.transforms:
            if isinstance(t, PasteBoxes):  # backgrounds: own rows without any animal
                empty = self.df[CLASSES].sum(axis=1) == 0
                t.backgrounds = [img_dir / f"{i}.jpg" for i in self.df.loc[empty, "image_id"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(
            Image.open(self.img_dir / f"{row['image_id']}.jpg").convert("RGB")
        )
        has_labels = set(CLASSES) <= set(row.index)
        vip = has_labels and bool(row[VIP].any())  # read by `NonVipDropout`, `NonVipImageCompression`, ignored by other transforms
        if self.boxes is None:
            image = self.transform(image=image, vip=vip)["image"]
        else:
            boxes = self.boxes.get(row["image_id"], [])
            image = self.transform(
                image=image,
                vip=vip,
                bboxes=np.array([b for _, b in boxes]).reshape(-1, 4),
                box_labels=[c for c, _ in boxes],
            )["image"]

        labels = (
            row[CLASSES].to_numpy(np.float32)
            if has_labels
            else np.zeros(len(CLASSES), np.float32)
        )
        return image, torch.from_numpy(labels)


def sample_weights(df, oversample=1, downsample=1.0):
    """Draws per epoch of each row of `df`: `oversample` for rows with any VIP label, `downsample` for others."""
    return np.where(df[VIP].any(axis=1), oversample, downsample)


def make_loader(ds, batch_size, num_workers, shuffle=False, sampler=None):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


class KaggleDataModule(pl.LightningDataModule):
    """Fold `fold` of `df` as validation, the rest as training. `fold=None`: train on all rows, no validation.
    `oversample` > 1: training rows with any VIP label drawn that many times per epoch (with replacement), others once.
    `downsample` < 1: other rows drawn that fraction of times per epoch, new random subset each epoch."""

    def __init__(
        self,
        df,
        fold=0,
        batch_size=32,
        num_workers=4,
        train_tfms=train_tfms(),
        val_tfms=val_tfms(),
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

    def setup(self, stage=None):
        is_val = self.df["fold"] == self.fold
        self.train_ds = MultilabelImageDataset(
            self.df[~is_val], TRAIN_DIR, self.train_tfms
        )
        self.val_ds = MultilabelImageDataset(self.df[is_val], TRAIN_DIR, self.val_tfms)

    def train_dataloader(self):
        if self.oversample == 1 and self.downsample == 1:
            return make_loader(
                self.train_ds, self.batch_size, self.num_workers, shuffle=True
            )
        # augmentation runs per draw in `__getitem__`, so every copy is new view
        weights = sample_weights(self.train_ds.df, self.oversample, self.downsample)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=round(weights.sum()), replacement=True
        )
        return make_loader(
            self.train_ds, self.batch_size, self.num_workers, sampler=sampler
        )

    def val_dataloader(self):
        if self.fold is None:
            return []
        return make_loader(self.val_ds, self.batch_size, self.num_workers)

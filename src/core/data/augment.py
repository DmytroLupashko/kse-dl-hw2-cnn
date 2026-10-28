"""Image transforms: training recipes as class tree, project transforms reading per-row `vip` flag.

YAML `data.augment` block (`class_path`) names recipe class: `KaggleDataModule` builds it with `img_size`, `.compose()` -> albumentations Compose; validation: `Validation(size).compose()`.
Subclass extends parent's transform list via `super().transforms()`. Dataset passes each row's `vip` flag.
"""
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from ..training.config import IMG_SIZE, MEAN, STD


class NonVipDropout(A.CoarseDropout):
    """`CoarseDropout` on rows without VIP label only: VIP rows get no holes."""

    def get_params_dependent_on_data(self, params, data):
        if data.get("vip"):
            return {"holes": np.empty((0, 4), np.int32), "seed": 0}

        return super().get_params_dependent_on_data(params, data)


class Recipe:
    """Image transform for `size` px images: `transforms()`, then Normalize and ToTensorV2; `compose_kw` passed to Compose."""

    def __init__(self, size=IMG_SIZE):
        self.size = size

    def compose(self, **compose_kw):
        return A.Compose([*self.transforms(), A.Normalize(MEAN, STD), ToTensorV2()], **compose_kw)


class Validation(Recipe):
    """Validation, test, Grad-CAM: resize only."""

    def transforms(self):
        return [A.Resize(self.size, self.size)]


class Default(Recipe):
    """Default augmentations (exp04)."""

    def transforms(self):
        return [
            A.Resize(self.size, self.size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
        ]


class CoarseDropoutNonVip(Recipe):
    """Default augmentations (exp04) with colored holes on non-VIP rows after flip (model exp20)."""

    def transforms(self):
        return [
            A.Resize(self.size, self.size),
            A.HorizontalFlip(p=0.5),
            # random color per hole: black would mimic Moris fur; VIP rows untouched, coat pattern stays (exp18)
            NonVipDropout(
                num_holes_range=(1, 4), hole_height_range=(0.1, 0.25), hole_width_range=(0.1, 0.25), fill="random_uniform", p=0.5
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
        ]

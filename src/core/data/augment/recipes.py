"""Training recipes as class tree: subclass extends parent's transform list via `super().transforms()`.

YAML `data.augment` block (`class_path`) names recipe class: `KaggleDataModule` builds it with `img_size`, `.compose()` -> albumentations Compose; validation: `Validation(size).compose()`.
"""
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from ...training.config import IMG_SIZE, MEAN, STD
from .ops import NonVipDropout, NonVipImageCompression, NonVipToGray


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


class Lecture(Recipe):
    """Training augmentations of Advanced CNN lecture (ResNet-18 transfer learning cell)."""

    def transforms(self):
        return [
            A.Resize(self.size, self.size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ]


class V2(Recipe):
    """Augmentations v2 (model exp05): orientation, zoom-in crop, color, one degradation, holes."""

    def transforms(self):
        size = self.size

        return [
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
        ]


class V3NoCrop(Recipe):
    """Augmentations v3 without crop (model exp07): flip, lighting, affine, image quality."""

    def transforms(self):
        return [
            A.Resize(self.size, self.size),
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
        ]


class GridDropout(Recipe):
    """Grid dropout (model exp18): default recipe plus grid dropout, `Affine` on every draw with zoom-out and shear."""

    def transforms(self):
        size = self.size

        return [
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


class Jpeg(CoarseDropoutNonVip):
    """exp20 recipe with JPEG quality 10 on half of training draws (model exp28)."""

    def transforms(self):
        # last before Normalize: model sees block artefacts as encoded; fur, whiskers, eyes lost first, outline kept
        return [*super().transforms(), A.ImageCompression(quality_range=(10, 10), p=0.5)]


class JpegNonVip(CoarseDropoutNonVip):
    """`Jpeg` on non-VIP rows only."""

    def transforms(self):
        return [*super().transforms(), NonVipImageCompression(quality_range=(10, 10), p=0.5)]


class GrayNonVip(CoarseDropoutNonVip):
    """exp20 recipe with gray on half of non-VIP training draws (model exp30)."""

    def transforms(self):
        # after holes: colored holes turn gray too, no color cue left on gray draws
        return [*super().transforms(), NonVipToGray(p=0.5)]

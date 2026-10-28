"""Box transforms: albumentations subclasses reading per-row pseudo-boxes (`bboxes`) and backgrounds."""

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from ..data.augment.ops import Transform
from .box import Box


class EraseBoxes(Transform, A.DualTransform):
    """Attention dropping: each pseudo-box filled independently with `p_box`, labels kept; fill zeros, or with
    `random_fill` one random color per box. Box larger than `max_area` of image skipped: likely whole animal, erasing it
    would leave label without animal. `skip_vip`: rows with VIP label untouched (`vip` flag, `RowTransform`)."""

    def __init__(
        self, p_box=0.5, max_area=0.3, random_fill=False, skip_vip=False, p=1.0
    ):
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
            if (x1 - x0) * (
                y1 - y0
            ) <= self.max_area and self.py_random.random() < self.p_box:
                color = (
                    [self.py_random.randint(0, 255) for _ in range(3)]
                    if self.random_fill
                    else 0
                )
                holes.append((*Box(x0, y0, x1, y1).pixels(w, h), color))

        return {"holes": holes}

    def apply(self, img, holes, **params):
        img = img.copy()

        for x0, y0, x1, y1, color in holes:
            img[y0:y1, x0:x1] = color

        return img

    def apply_to_bboxes(self, bboxes, **params):
        return bboxes


class PasteBoxes(Transform, A.DualTransform):
    """Background paste (CMO, model exp17): with `p`, union of pseudo-boxes plus `margin` cut out and pasted at random
    place onto random background, longer side `scale` of background's shorter side; labels kept.
    `backgrounds`: image paths, set by `BoxExtras.bind` to its rows without labels (train part only)."""

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
        crop = Box(max(x0, 0), max(y0, 0), min(x1, 1), min(y1, 1)).pixels(w, h)
        side = self.py_random.uniform(*self.scale) * min(h, w)
        k = side / max(crop[2] - crop[0], crop[3] - crop[1])
        size = (
            max(int((crop[2] - crop[0]) * k), 1),
            max(int((crop[3] - crop[1]) * k), 1),
        )  # (width, height)
        at = (
            self.py_random.randint(0, w - size[0]),
            self.py_random.randint(0, h - size[1]),
        )

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

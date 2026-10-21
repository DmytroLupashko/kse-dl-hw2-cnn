"""Box geometry in [0, 1] image coordinates (albumentations format): pseudo-boxes, erase probes."""
from typing import NamedTuple

import numpy as np

from .config import MEAN


class Box(NamedTuple):
    """(x_min, y_min, x_max, y_max) in [0, 1]; plain tuple wherever tuple expected (albumentations, CSV rows)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_cam(cls, cam, threshold):
        """Bounding box of Grad-CAM pixels at or above `threshold`; None when map empty."""
        ys, xs = np.nonzero(cam >= threshold)
        if cam.max() <= 0 or len(xs) == 0:
            return None
        h, w = cam.shape
        return cls(xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)

    def grow(self, factor):
        """Box scaled `factor` times in width and height around its center, clipped to image
        (model exp23: Cat face core to whole head)."""
        x0, y0, x1, y1 = self
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
        return Box(max(cx - hw, 0.0), max(cy - hh, 0.0), min(cx + hw, 1.0), min(cy + hh, 1.0))

    def head(self):
        """Cat face core grown to head and ears (model exp24): 2x wide around center, top raised by 1x height
        (ears sit above core), bottom lowered by 0.25x height, clipped to image."""
        x0, y0, x1, y1 = self
        w, h = x1 - x0, y1 - y0
        return Box(max(x0 - w / 2, 0.0), max(y0 - h, 0.0), min(x1 + w / 2, 1.0), min(y1 + h / 4, 1.0))

    def elsewhere(self, rng, tries=100):
        """Box of same size at random place not overlapping this one; last try when none fits."""
        x0, y0, x1, y1 = self
        w, h = x1 - x0, y1 - y0
        for _ in range(tries):
            nx, ny = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            if nx + w <= x0 or nx >= x1 or ny + h <= y0 or ny >= y1:
                break
        return Box(nx, ny, nx + w, ny + h)

    def erase(self, image):
        """Copy of HWC uint8 `image` with box filled with normalization mean."""
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self
        image = image.copy()
        image[int(y0 * h) : int(np.ceil(y1 * h)), int(x0 * w) : int(np.ceil(x1 * w))] = np.array(MEAN) * 255
        return image

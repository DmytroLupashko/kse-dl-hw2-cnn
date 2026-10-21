"""Per-row extras for transforms: pseudo-boxes from file, VIP flag, backgrounds."""
import numpy as np
import pandas as pd

from ..box import Box
from ..config import CLASSES, VIP
from .ops import PasteBoxes
from .recipes import BOXES


def read_boxes(path=BOXES):
    """{image_id: [(class, `Box`), ...]} from `path`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m core.boxes`")
    df = pd.read_csv(path)
    coords = df[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    boxes = {}
    for image_id, cls, box in zip(df["image_id"], df["cls"], coords):
        boxes.setdefault(image_id, []).append((cls, Box(*box)))
    return boxes


class RowTransform:
    """`transform` bound to rows of `df` (images in `img_dir`); call `(image, row)` -> transformed image.
    Feeds extras its transforms read: `vip` flag, pseudo-boxes of row, backgrounds (rows of `df` without labels).
    Class, not closure: DataLoader workers pickle it."""

    def __init__(self, transform, df, img_dir):
        self.transform = transform
        # transform with `bbox_params` (v3) gets each image's pseudo-boxes
        self.boxes = read_boxes(transform.boxes_path) if "bboxes" in transform.processors else None
        for t in transform.transforms:
            if isinstance(t, PasteBoxes):  # backgrounds: own rows without any animal
                empty = df[CLASSES].sum(axis=1) == 0
                t.backgrounds = [img_dir / f"{i}.jpg" for i in df.loc[empty, "image_id"]]

    def __call__(self, image, row):
        has_labels = set(CLASSES) <= set(row.index)
        vip = has_labels and bool(row[VIP].any())  # read by `NonVipDropout`, `NonVipImageCompression`, ignored by other transforms
        if self.boxes is None:
            return self.transform(image=image, vip=vip)["image"]
        boxes = self.boxes.get(row["image_id"], [])
        return self.transform(
            image=image,
            vip=vip,
            bboxes=np.array([b for _, b in boxes]).reshape(-1, 4),
            box_labels=[c for c, _ in boxes],
        )["image"]

"""Box files and per-row extras of box recipes: pseudo-boxes of row, backgrounds for `PasteBoxes`."""

import numpy as np
import pandas as pd

from ..training.config import CLASSES, PROCESSED
from .box import Box
from .ops import PasteBoxes

BOXES = (
    PROCESSED / "boxes.csv"
)  # Grad-CAM pseudo-boxes of training images, written by `python -m core.boxes.pseudo`
FACE_BOXES = (
    PROCESSED / "boxes_face.csv"
)  # Grad-CAM hot core of Cat, Dog (model exp21), `python -m core.boxes.pseudo --kind=face`
HEAD_BOXES = (
    PROCESSED / "boxes_head.csv"
)  # face boxes, Cat grown to head and ears (model exp24), `python -m core.boxes.pseudo --kind=head`


def read_boxes(path=BOXES):
    """{image_id: [(class, `Box`), ...]} from `path` (written by `python -m core.boxes.pseudo`)."""
    df = pd.read_csv(path)
    coords = df[list(Box._fields)].to_numpy()
    boxes = {}

    for image_id, cls, box in zip(df["image_id"], df["cls"], coords):
        boxes.setdefault(image_id, []).append((cls, Box(*box)))

    return boxes


class BoxExtras:
    """`row_extras` of box recipe Compose, read by `RowTransform`: `bind(transform, df, img_dir)` loads pseudo-boxes of
    `path` and gives `PasteBoxes` backgrounds (rows of `df` without labels); call `(row)` -> `bboxes`, `box_labels` of row.
    Class, not closure: DataLoader workers pickle it."""

    def __init__(self, path):
        self.path = path
        self.boxes = {}

    def bind(self, transform, df, img_dir):
        self.boxes = read_boxes(self.path)

        for t in transform.transforms:
            if isinstance(t, PasteBoxes):  # backgrounds: own rows without any animal
                empty = df[CLASSES].sum(axis=1) == 0
                t.backgrounds = [
                    img_dir / f"{i}.jpg" for i in df.loc[empty, "image_id"]
                ]

    def __call__(self, row):
        boxes = self.boxes.get(row["image_id"], [])

        return {
            "bboxes": np.array([b for _, b in boxes]).reshape(-1, 4),
            "box_labels": [c for c, _ in boxes],
        }

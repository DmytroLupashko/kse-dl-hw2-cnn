"""Grad-CAM pseudo-boxes for `BBoxSafeRandomCrop` (`core.data.v3_tfms`), no manual annotation.

Per training image, one box per present class: bounding rectangle of Grad-CAM map >= `CAM_THRESHOLD` of its max,
by exp04 ResNet-18 (fold model that held image out, else fold 0). Image without labels: box of its top-scoring class.
CLI: `python -m core.boxes`, writes `core.data.BOXES` (albumentations format, coordinates in [0, 1]).
Face boxes (model exp21): `python -m core.boxes --face`, writes `core.data.FACE_BOXES`: hot core (>= `FACE_THRESHOLD`)
of present Cat, Dog only, by exp20 ResNet-18.
Head boxes (model exp24): `python -m core.boxes --head`, writes `core.data.HEAD_BOXES`: face boxes with Cat ones grown to
head and ears (`head_box`), Dog ones unchanged; needs face boxes, no new Grad-CAM.
"""
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import CLASSES, ROOT, TRAIN_DIR
from .cv import DEVICE, Config
from .data import BOXES, FACE_BOXES, HEAD_BOXES, read_train, val_tfms
from .gradcam import grad_cam

SOURCE = ROOT / "sections/model/configs/exp04_oversampling_vip/resnet18.toml"
CAM_THRESHOLD = 0.5
FACE_SOURCE = ROOT / "sections/model/configs/exp20_coarse_dropout_non_vip/resnet18.toml"
FACE_THRESHOLD = 0.8  # 0.5 boxes cover whole animal (exp14); core of map is face on Cat, Dog (exp19)
BATCH = 32


def cam_box(cam, threshold=CAM_THRESHOLD):
    """(x_min, y_min, x_max, y_max) in [0, 1] of pixels at or above `threshold`; None when map empty."""
    ys, xs = np.nonzero(cam >= threshold)
    if cam.max() <= 0 or len(xs) == 0:
        return None
    h, w = cam.shape
    return xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h


def grow(box, factor):
    """`box` (x_min, y_min, x_max, y_max in [0, 1]) scaled `factor` times in width and height around its center,
    clipped to image (model exp23: Cat face core to whole head)."""
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
    return max(cx - hw, 0.0), max(cy - hh, 0.0), min(cx + hw, 1.0), min(cy + hh, 1.0)


def head_box(box):
    """Cat face core `box` grown to head and ears (model exp24): 2x wide around center, top raised by 1x height
    (ears sit above core), bottom lowered by 0.25x height, clipped to image."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return max(x0 - w / 2, 0.0), max(y0 - h, 0.0), min(x1 + w / 2, 1.0), min(y1 + h / 4, 1.0)


def pseudo_boxes(df, config, threshold=CAM_THRESHOLD, classes=None):
    """Box rows (image_id, cls, x_min, y_min, x_max, y_max) for rows of `df` (`read_train()` table).
    `classes` set: boxes of present `classes` only, none for images without labels."""
    models = {fold: config.fold_model(fold).model for fold in config.folds}
    tfms = val_tfms(config.img_size)
    rows = []
    for fold, part in df.groupby(df["fold"].where(df["fold"].isin(models.keys()), config.folds[0])):
        model = models[fold]
        for start in tqdm(range(0, len(part), BATCH), desc=f"model fold {fold}"):
            batch = part.iloc[start : start + BATCH]
            x = torch.stack(
                [tfms(image=np.array(Image.open(TRAIN_DIR / f"{i}.jpg").convert("RGB")))["image"] for i in batch["image_id"]]
            ).to(DEVICE)
            with torch.no_grad():
                top = model.eval()(x).argmax(1).cpu().numpy()
            labels = batch[CLASSES].to_numpy() == 1
            for c, cls in enumerate(CLASSES):
                if classes is None:
                    wanted = labels[:, c] | (~labels.any(1) & (top == c))
                elif cls in classes:
                    wanted = labels[:, c]
                else:
                    continue
                if not wanted.any():
                    continue
                cams = grad_cam(model, model.layer4, x[wanted], c)
                for image_id, cam in zip(batch["image_id"][wanted], cams):
                    if (box := cam_box(cam, threshold)) is not None:
                        rows.append((image_id, cls, *box))
    return pd.DataFrame(rows, columns=["image_id", "cls", "x_min", "y_min", "x_max", "y_max"])


if __name__ == "__main__":
    if "--head" in sys.argv[1:]:
        path = HEAD_BOXES
        boxes = pd.read_csv(FACE_BOXES)
        cat = boxes["cls"] == "Cat"
        coords = ["x_min", "y_min", "x_max", "y_max"]
        boxes.loc[cat, coords] = [head_box(b) for b in boxes.loc[cat, coords].itertuples(index=False)]
    elif "--face" in sys.argv[1:]:
        path = FACE_BOXES
        boxes = pseudo_boxes(read_train(), Config.from_toml(FACE_SOURCE), FACE_THRESHOLD, classes=("Cat", "Dog"))
    else:
        path = BOXES
        boxes = pseudo_boxes(read_train(), Config.from_toml(SOURCE))
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes.to_csv(path, index=False)
    print(f"{len(boxes)} boxes, {boxes['image_id'].nunique()} images: {path}")

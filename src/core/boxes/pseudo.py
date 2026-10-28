"""Grad-CAM pseudo-boxes for `BBoxSafeRandomCrop` (`core.boxes.V3`), no manual annotation.

Per training image, one box per present class: bounding rectangle of Grad-CAM map >= `CAM_THRESHOLD` of its max,
by exp04 ResNet-18 (fold model that held image out, else fold 0). Image without labels: box of its top-scoring class.
CLI: `python -m core.boxes.pseudo`, writes `core.boxes.BOXES` (albumentations format, coordinates in [0, 1]).
Face boxes (model exp21): `python -m core.boxes.pseudo --kind=face`, writes `core.boxes.FACE_BOXES`: hot core (>= `FACE_THRESHOLD`)
of present Cat, Dog only, by exp20 ResNet-18.
Head boxes (model exp24): `python -m core.boxes.pseudo --kind=head`, writes `core.boxes.HEAD_BOXES`: face boxes with Cat ones grown to
head and ears (`Box.head`), Dog ones unchanged; needs face boxes, no new Grad-CAM.
"""

from typing import Literal

import numpy as np
import pandas as pd
import torch
from jsonargparse import auto_cli
from PIL import Image
from tqdm import tqdm

from ..analysis.gradcam import GradCam
from ..data.augment import Validation
from ..data.dataset import read_train
from ..training.config import CLASSES, DEVICE, ROOT, TRAIN_DIR
from ..training.cv import CVResult
from .box import Box
from .rows import BOXES, FACE_BOXES, HEAD_BOXES

SOURCE = ROOT / "configs/models/exp04_oversampling_vip/resnet18.yaml"
CAM_THRESHOLD = 0.5
FACE_SOURCE = ROOT / "configs/models/exp20_coarse_dropout_non_vip/resnet18.yaml"
FACE_THRESHOLD = (
    0.8  # 0.5 boxes cover whole animal (exp14); core of map is face on Cat, Dog (exp19)
)
BATCH = 32


def pseudo_boxes(config, df, threshold=CAM_THRESHOLD, classes=None):
    """Grad-CAM boxes of fold models of model config path `config` for rows of `df` (`read_train()` table), by fold model that held image
    out, else first fold: per image, `Box.from_cam` at `threshold` for each present class. `classes` set: boxes of
    present `classes` only, none for images without labels; else image without labels gets box of its top-scoring class.
    Rows: `image_id`, `cls`, `Box` fields."""
    result = CVResult.load(config)
    cams = {fold: GradCam(result.fold_model(fold).model) for fold in result.folds}
    tfms = Validation(result.config["data"]["img_size"]).compose()
    rows = []

    for fold, part in df.groupby(
        df["fold"].where(df["fold"].isin(cams.keys()), result.folds[0])
    ):
        cam = cams[fold]

        for start in tqdm(range(0, len(part), BATCH), desc=f"model fold {fold}"):
            batch = part.iloc[start : start + BATCH]
            x = torch.stack(
                [
                    tfms(
                        image=np.array(
                            Image.open(TRAIN_DIR / f"{i}.jpg").convert("RGB")
                        )
                    )["image"]
                    for i in batch["image_id"]
                ]
            ).to(DEVICE)
            labels = batch[CLASSES].to_numpy() == 1

            if classes is None:
                with torch.no_grad():
                    top = cam.model.eval()(x).argmax(1).cpu().numpy()

            for c, cls in enumerate(CLASSES):
                if classes is None:
                    wanted = labels[:, c] | (~labels.any(1) & (top == c))
                elif cls in classes:
                    wanted = labels[:, c]
                else:
                    continue

                if not wanted.any():
                    continue

                for image_id, m in zip(batch["image_id"][wanted], cam(x[wanted], c)):
                    if (box := Box.from_cam(m, threshold)) is not None:
                        rows.append((image_id, cls, *box))

    return pd.DataFrame(rows, columns=["image_id", "cls", *Box._fields])


def main(kind: Literal["body", "face", "head"] = "body"):
    if kind == "head":
        path = HEAD_BOXES
        boxes = pd.read_csv(FACE_BOXES)
        cat = boxes["cls"] == "Cat"
        coords = list(Box._fields)
        boxes.loc[cat, coords] = [
            Box(*b).head() for b in boxes.loc[cat, coords].itertuples(index=False)
        ]
    elif kind == "face":
        path = FACE_BOXES
        boxes = pseudo_boxes(
            FACE_SOURCE, read_train(), FACE_THRESHOLD, classes=("Cat", "Dog")
        )
    else:
        path = BOXES
        boxes = pseudo_boxes(SOURCE, read_train())

    path.parent.mkdir(parents=True, exist_ok=True)
    boxes.to_csv(path, index=False)
    print(f"{len(boxes)} boxes, {boxes['image_id'].nunique()} images: {path}")


if __name__ == "__main__":
    auto_cli(main)

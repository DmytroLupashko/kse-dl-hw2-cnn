"""Grad-CAM pseudo-boxes for `BBoxSafeRandomCrop` (`core.augment.V3`), no manual annotation.

Per training image, one box per present class: bounding rectangle of Grad-CAM map >= `CAM_THRESHOLD` of its max,
by exp04 ResNet-18 (fold model that held image out, else fold 0). Image without labels: box of its top-scoring class.
CLI: `python -m core.boxes`, writes `core.augment.BOXES` (albumentations format, coordinates in [0, 1]).
Face boxes (model exp21): `python -m core.boxes --face`, writes `core.augment.FACE_BOXES`: hot core (>= `FACE_THRESHOLD`)
of present Cat, Dog only, by exp20 ResNet-18.
Head boxes (model exp24): `python -m core.boxes --head`, writes `core.augment.HEAD_BOXES`: face boxes with Cat ones grown to
head and ears (`Box.head`), Dog ones unchanged; needs face boxes, no new Grad-CAM.
"""
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .augment import BOXES, FACE_BOXES, HEAD_BOXES, Validation
from .box import Box
from .config import CLASSES, DEVICE, ROOT, TRAIN_DIR
from .cv import Config
from .data import read_train
from .gradcam import GradCam

SOURCE = ROOT / "sections/model/configs/exp04_oversampling_vip/resnet18.toml"
CAM_THRESHOLD = 0.5
FACE_SOURCE = ROOT / "sections/model/configs/exp20_coarse_dropout_non_vip/resnet18.toml"
FACE_THRESHOLD = 0.8  # 0.5 boxes cover whole animal (exp14); core of map is face on Cat, Dog (exp19)
BATCH = 32


class PseudoBoxes:
    """Grad-CAM boxes of fold models of `config`: per image, `Box.from_cam` at `threshold` for each present class.
    `classes` set: boxes of present `classes` only, none for images without labels; else image without labels gets box
    of its top-scoring class."""

    COLUMNS = ["image_id", "cls", "x_min", "y_min", "x_max", "y_max"]

    def __init__(self, config, threshold=CAM_THRESHOLD, classes=None):
        self.config = config
        self.threshold = threshold
        self.classes = classes

    def wanted(self, labels, top, c, cls):
        """Mask of batch rows that get box of class `c` (`cls`); None: class skipped."""
        if self.classes is None:
            return labels[:, c] | (~labels.any(1) & (top == c))
        return labels[:, c] if cls in self.classes else None

    def compute(self, df):
        """Box rows (`COLUMNS`) for rows of `df` (`read_train()` table): fold model that held image out, else first fold."""
        config = self.config
        models = {fold: config.fold_model(fold).model for fold in config.folds}
        tfms = Validation(config.img_size).compose()
        rows = []
        for fold, part in df.groupby(df["fold"].where(df["fold"].isin(models.keys()), config.folds[0])):
            model = models[fold]
            cam = GradCam(model)
            for start in tqdm(range(0, len(part), BATCH), desc=f"model fold {fold}"):
                batch = part.iloc[start : start + BATCH]
                x = torch.stack(
                    [tfms(image=np.array(Image.open(TRAIN_DIR / f"{i}.jpg").convert("RGB")))["image"] for i in batch["image_id"]]
                ).to(DEVICE)
                with torch.no_grad():
                    top = model.eval()(x).argmax(1).cpu().numpy()
                labels = batch[CLASSES].to_numpy() == 1
                for c, cls in enumerate(CLASSES):
                    wanted = self.wanted(labels, top, c, cls)
                    if wanted is None or not wanted.any():
                        continue
                    for image_id, m in zip(batch["image_id"][wanted], cam(x[wanted], c)):
                        if (box := Box.from_cam(m, self.threshold)) is not None:
                            rows.append((image_id, cls, *box))
        return pd.DataFrame(rows, columns=self.COLUMNS)


if __name__ == "__main__":
    if "--head" in sys.argv[1:]:
        path = HEAD_BOXES
        boxes = pd.read_csv(FACE_BOXES)
        cat = boxes["cls"] == "Cat"
        coords = ["x_min", "y_min", "x_max", "y_max"]
        boxes.loc[cat, coords] = [Box(*b).head() for b in boxes.loc[cat, coords].itertuples(index=False)]
    elif "--face" in sys.argv[1:]:
        path = FACE_BOXES
        boxes = PseudoBoxes(Config.from_toml(FACE_SOURCE), FACE_THRESHOLD, classes=("Cat", "Dog")).compute(read_train())
    else:
        path = BOXES
        boxes = PseudoBoxes(Config.from_toml(SOURCE)).compute(read_train())
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes.to_csv(path, index=False)
    print(f"{len(boxes)} boxes, {boxes['image_id'].nunique()} images: {path}")

"""Box recipes (`DataConfig.augment` `_target_`): main recipes with box transforms in front, Compose fed pseudo-boxes."""

import albumentations as A

from ..data.augment.recipes import CoarseDropoutNonVip, Default, V3NoCrop
from .ops import EraseBoxes, PasteBoxes
from .rows import BOXES, FACE_BOXES, HEAD_BOXES, BoxExtras


class BoxRecipe:
    """Mixin before main recipe: Compose takes pseudo-boxes of `boxes` file (`bboxes`, `box_labels`) from `BoxExtras`."""

    boxes = BOXES

    def compose(self):
        compose = super().compose(
            bbox_params=A.BboxParams(
                format="albumentations", label_fields=["box_labels"], clip=True
            )
        )
        compose.row_extras = BoxExtras(self.boxes)

        return compose


class V3(BoxRecipe, V3NoCrop):
    """Augmentations v3 (model exp06): box-safe crop, every pseudo-box kept whole, plus `V3NoCrop`."""

    def transforms(self):
        return [A.BBoxSafeRandomCrop(erosion_rate=0.0, p=1), *super().transforms()]


class BoxCrop(BoxRecipe, Default):
    """Default augmentations (exp04) with box-safe crop in front, every pseudo-box kept whole (model exp07)."""

    def transforms(self):
        return [A.BBoxSafeRandomCrop(erosion_rate=0.0, p=1), *super().transforms()]


class AttentionDrop(BoxRecipe, Default):
    """Default augmentations (exp04) with pseudo-box erase in front (model exp14)."""

    def transforms(self):
        return [EraseBoxes(), *super().transforms()]


class Paste(BoxRecipe, Default):
    """Default augmentations (exp04) with background paste in front (model exp17)."""

    def transforms(self):
        return [PasteBoxes(), *super().transforms()]


class FaceErase(BoxRecipe, CoarseDropoutNonVip):
    """exp20 recipe with Cat, Dog boxes of `boxes` erased on non-VIP rows in front (model exp21: face boxes)."""

    boxes = FACE_BOXES

    def transforms(self):
        # random color: black would mimic Moris fur
        return [EraseBoxes(random_fill=True, skip_vip=True), *super().transforms()]


class HeadErase(FaceErase):
    """exp21 recipe with Cat head boxes instead of face boxes (model exp24)."""

    boxes = HEAD_BOXES

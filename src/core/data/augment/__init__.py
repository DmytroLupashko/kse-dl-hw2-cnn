"""Image transforms: training recipes, validation, per-row extras.

`recipes`: `Recipe` class tree, `RECIPES` registry (`Config.augment` key), pseudo-box files.
`ops`: project albumentations transforms (box erase, background paste, non-VIP variants).
`rows`: `RowTransform` feeds each image its extras (`vip` flag, pseudo-boxes, backgrounds); dataset only loads images.
"""
from .recipes import (
    BOXES,
    FACE_BOXES,
    HEAD_BOXES,
    RECIPES,
    AttentionDrop,
    BoxCrop,
    CoarseDropoutNonVip,
    Default,
    FaceErase,
    GrayNonVip,
    GridDropout,
    HeadErase,
    Jpeg,
    JpegNonVip,
    Lecture,
    Paste,
    Recipe,
    V2,
    V3,
    V3NoCrop,
    Validation,
)
from .rows import RowTransform, read_boxes

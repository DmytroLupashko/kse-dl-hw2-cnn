"""Image transforms: training recipes, validation, per-row extras.

`recipes`: `Recipe` class tree (YAML `data.augment` `class_path`).
`ops`: project albumentations transforms (non-VIP variants).
`rows`: `RowTransform` feeds each image its extras (`vip` flag, Compose `row_extras`); dataset only loads images.
Box recipes and transforms: `core.boxes`.
"""
from .recipes import V2, Default, Validation
from .rows import RowTransform

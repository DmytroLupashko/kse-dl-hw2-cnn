"""Per-row extras for transforms: VIP flag, `row_extras` of Compose."""
from ...training.config import CLASSES, VIP


class RowTransform:
    """`transform` bound to rows of `df` (images in `img_dir`); call `(image, row)` -> transformed image.
    Feeds extras its transforms read: `vip` flag; Compose with `row_extras` (`bind(transform, df, img_dir)`, then
    `(row)` -> dict of extra inputs) adds its own. Class, not closure: DataLoader workers pickle it."""

    def __init__(self, transform, df, img_dir):
        self.transform = transform
        self.extras = getattr(transform, "row_extras", None)

        if self.extras is not None:
            self.extras.bind(transform, df, img_dir)

    def __call__(self, image, row):
        has_labels = set(CLASSES) <= set(row.index)
        # read by `NonVipDropout`, `NonVipImageCompression`, `NonVipToGray`, `skip_vip` transforms; ignored by others
        vip = has_labels and bool(row[VIP].any())
        extras = {} if self.extras is None else self.extras(row)

        return self.transform(image=image, vip=vip, **extras)["image"]

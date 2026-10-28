"""Project transforms: albumentations subclasses reading per-row `vip` flag."""
import albumentations as A
import numpy as np


class Transform:
    """Mixin of every project transform. Serialized name `core.data.<class>`: module where finished runs were trained;
    recipe hash covers it, so finished runs stay reused wherever class lives."""

    @classmethod
    def get_class_fullname(cls):
        return f"core.data.{cls.__name__}"


class NonVip(Transform):
    """Transform skipped on rows with VIP label; `vip` flag passed by `RowTransform`."""

    def get_params_dependent_on_data(self, params, data):
        return {**super().get_params_dependent_on_data(params, data), "vip": bool(data.get("vip"))}

    def apply(self, img, *args, vip=False, **params):
        return img if vip else super().apply(img, *args, **params)


class NonVipImageCompression(NonVip, A.ImageCompression):
    """`ImageCompression` on rows without VIP label only."""


class NonVipToGray(NonVip, A.ToGray):
    """`ToGray` on rows without VIP label only."""


class NonVipDropout(Transform, A.CoarseDropout):
    """`CoarseDropout` on rows without VIP label only. VIP rows get no holes, not skipped apply: boxes need `holes` too."""

    def get_params_dependent_on_data(self, params, data):
        if data.get("vip"):
            return {"holes": np.empty((0, 4), np.int32), "seed": 0}

        return super().get_params_dependent_on_data(params, data)

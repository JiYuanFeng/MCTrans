import numpy as np
from typing import Mapping, Hashable, Dict

from mmcv.utils import build_from_cfg
from monai.config import KeysCollection
from monai.transforms import apply_transform, MapTransform

from ..builder import TRANSFORMS


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = apply_transform(t, data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def build_transforms(cfg):
    """Build a transformer.

    Args:
        cfg (dict, list[dict]): The config of tranforms, is is either a dict
            or a list of configs.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        transforms = [
            build_from_cfg(cfg_, TRANSFORMS) for cfg_ in cfg
        ]
        return Compose(transforms)
    else:
        return build_from_cfg(cfg, TRANSFORMS)


@TRANSFORMS.register_module()
class BinrayLabel(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False,
                 ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key][d[key] > 0] = 1
        return d

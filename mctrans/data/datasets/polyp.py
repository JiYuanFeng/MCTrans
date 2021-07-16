import os.path as osp
from .base import BaseDataset
from ..builder import DATASETS


@DATASETS.register_module()
class LesionDataset(BaseDataset):
    CLASSES = ('background', 'lesion')
    PALETTE = [[120, 120, 120], [180, 120, 120]]

    def __init__(self, **kwargs):
        super(LesionDataset, self).__init__(
            label_map={255: 1}, **kwargs)
        assert osp.exists(self.img_dir) and self.phase is not None

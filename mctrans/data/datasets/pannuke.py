from .base import BaseDataset
from ..builder import DATASETS

@DATASETS.register_module()
class PanNukeDataset(BaseDataset):
    CLASSES = ('Background', 'Neoplastic', "Inflammatory", "Connective", "Dead", "Non-Neoplastic Epithelial")
    def __init__(self, **kwargs):
        super(PanNukeDataset, self).__init__(**kwargs)

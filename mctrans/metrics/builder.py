from mmcv.utils import Registry, build_from_cfg
from abc import ABC, abstractmethod

METRICS = Registry('metric')

def build_metrics(cfg):
    """Build metric."""
    class_names = cfg.class_names
    cfg_types = cfg.metric_types
    metrics = [build_from_cfg(_cfg, METRICS) for _cfg in cfg_types]
    for metric in metrics:
        metric.set_class_name(class_names)
    return metrics

class Metric(ABC):

    @abstractmethod
    def __call__(self, pred, target, *args, **kwargs):
        raise NotImplementedError
    
    def set_class_name(self, class_names):
        self.class_names = class_names


from mmcv.utils import Registry, build_from_cfg
from torch import nn

NETWORKS = Registry('network')
LOSSES = Registry('loss')
MODEL = Registry('model')
ENCODERS = Registry('encoder')
DECODERS = Registry('decoder')
CENTERS = Registry('center')
HEADS = Registry('head')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_network(cfg):
    """Build network."""
    return build(cfg, NETWORKS)


def build_losses(cfg):
    """Build loss."""
    return [build_from_cfg(_cfg, LOSSES) for _cfg in cfg]


def build_model(cfg):
    """Build model."""
    return build(cfg, MODEL)


def build_encoder(cfg):
    """Build Encoder."""
    return build(cfg, ENCODERS)


def build_decoder(cfg):
    """Build Decoder."""
    return build(cfg, DECODERS)


def build_center(cfg):
    """Build Center."""
    return build(cfg, CENTERS)


def build_head(cfg):
    """Build SegHead."""
    return build(cfg, HEADS)

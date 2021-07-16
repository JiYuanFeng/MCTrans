import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import ENCODERS
from ..utils import make_vgg_layer


@ENCODERS.register_module()
class VGG(nn.Module):
    def __init__(self, in_channel=1, depth=5, init_channels=16, num_blocks=2):
        super(VGG, self).__init__()
        filters = [(2 ** i) * init_channels for i in range(depth)]
        self.out_channels = filters.copy()

        filters.insert(0, in_channel)
        self.stages = nn.ModuleList()

        for idx in range(depth):
            down_sample = False if idx == 0 else True
            self.stages.append(make_vgg_layer(inplanes=filters[idx],
                                              planes=filters[idx + 1],
                                              num_blocks=num_blocks,
                                              with_bn=True,
                                              down_sample=down_sample))

    def forward(self, x):

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def init_weights(self, pretrained=None):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         kaiming_init(m)
        #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
        #         constant_init(m, 1)

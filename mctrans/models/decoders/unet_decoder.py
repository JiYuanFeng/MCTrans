import torch
import torch.nn as nn

from ..builder import DECODERS
from ..utils import conv_bn_relu


class AttBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels=F_g,
                      out_channels=F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels=F_l,
                      out_channels=F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=F_int,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DecBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            attention=False
    ):
        super().__init__()
        self.conv1 = conv_bn_relu(in_channels=in_channels + skip_channels,
                                  out_channels=out_channels)

        self.conv2 = conv_bn_relu(in_channels=out_channels,
                                  out_channels=out_channels)

        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        if attention:
            self.att = AttBlock(F_g=in_channels, F_l=skip_channels, F_int=in_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if hasattr(self, "att"):
                skip = self.att(g=x, x=skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


@DECODERS.register_module()
class UNetDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            att=False
    ):
        super().__init__()
        self.decoders = nn.ModuleList()
        in_channels = in_channels[::-1]
        skip_channels = in_channels[1:]
        for in_c, skip_c in zip(in_channels, skip_channels):
            self.decoders.append(DecBlock(in_c, skip_c, skip_c, att))

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        skips = features[1:]

        for i, layer in enumerate(self.decoders):
            x = layer(x, skips[i])

        return x

    def init_weights(self):
        pass


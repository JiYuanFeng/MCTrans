import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import CENTERS


class DacBlock(nn.Module):
    def __init__(self, channel):
        super(DacBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.conv1x1(self.dilate2(x)))
        dilate3_out = self.relu(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = self.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SppBlock(nn.Module):
    def __init__(self, in_channels):
        super(SppBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


@CENTERS.register_module()
class CEncoder(nn.Module):
    def __init__(self,
                 in_channels=[240],
                 ):
        super().__init__()
        in_channels = in_channels[-1]
        self.dac = DacBlock(in_channels)
        self.spp = SppBlock(in_channels)

    def forward(self, x):
        feat = x[-1]
        feat = self.dac(feat)
        feat = self.spp(feat)
        x[-1] = feat
        return x

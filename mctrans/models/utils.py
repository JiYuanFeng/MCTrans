import torch.nn as nn


def conv3x3(in_planes, out_planes, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation)


def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


def deconv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, stride=2):
    return nn.Sequential(nn.ConvTranspose2d(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            stride=stride,
                                            output_padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


def make_vgg_layer(inplanes,
                   planes,
                   num_blocks,
                   dilation=1,
                   with_bn=False,
                   down_sample=False,
                   ceil_mode=False):
    layers = []
    if down_sample:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    return nn.Sequential(*layers)

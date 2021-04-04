""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class DoubleConv(nn.Module):
    """(convolution => [BN] => Tanh) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, padding_size=1):
        super().__init__()
        if kernel_size == 3:
            padding_size = 1
        if kernel_size == 1:
            padding_size = 0
        layers = []
        for _ in range(2):
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding_size, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()]
            in_channels = out_channels
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


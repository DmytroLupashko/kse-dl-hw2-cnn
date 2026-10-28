"""Own ResNet-34 from He et al. 2015 (assignment item "implement your own CNN"). YAML `model.net` block:
`class_path: core.training.resnet_paper.ResnetFromPaper`, `num_classes` in `init_args`."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv, BatchNorm (`use_batchnorm`, else Identity), ReLU, then 3x3 stride-2 max pool when `pool`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        pool=False,
        use_batchnorm=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class ResnetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))


# I did
class ResnetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, initial=False):
        super().__init__()
        conv1_stride = 2 if initial else 1
        self.conv1 = ResnetConv(
            in_channels,
            out_channels,
            3,
            stride=conv1_stride,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = ResnetConv(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcutConv = (
            ResnetConv(
                in_channels,
                out_channels,
                1,
                stride=conv1_stride,
                use_batchnorm=use_batchnorm,
            )
            if initial
            else nn.Identity()
        )

    def forward(self, x):
        y1 = self.relu1(self.conv1(x))
        y2 = self.conv2(y1)
        x1 = self.shortcutConv(x)

        return self.relu2(y2 + x1)


# I did, unfinished
class ResnetFromPaper(nn.Module):
    # By default Paper uses batchnorm
    # It does not use dropout
    def __init__(self, num_classes, use_batchnorm=True):
        super().__init__()

        self.inputConv = ConvBlock(
            3, 64, 7, stride=2, padding=3, use_batchnorm=use_batchnorm
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnetBlock64_1 = ResnetConvBlock(64, 64, use_batchnorm=use_batchnorm)
        self.resnetBlock64_2 = ResnetConvBlock(64, 64, use_batchnorm=use_batchnorm)
        self.resnetBlock64_3 = ResnetConvBlock(64, 64, use_batchnorm=use_batchnorm)

        self.resnetBlock128_1 = ResnetConvBlock(
            64, 128, use_batchnorm=use_batchnorm, initial=True
        )
        self.resnetBlock128_2 = ResnetConvBlock(128, 128, use_batchnorm=use_batchnorm)
        self.resnetBlock128_3 = ResnetConvBlock(128, 128, use_batchnorm=use_batchnorm)
        self.resnetBlock128_4 = ResnetConvBlock(128, 128, use_batchnorm=use_batchnorm)

        self.resnetBlock256_1 = ResnetConvBlock(
            128, 256, use_batchnorm=use_batchnorm, initial=True
        )
        self.resnetBlock256_2 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_3 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_4 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_5 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)
        self.resnetBlock256_6 = ResnetConvBlock(256, 256, use_batchnorm=use_batchnorm)

        self.resnetBlock512_1 = ResnetConvBlock(
            256, 512, use_batchnorm=use_batchnorm, initial=True
        )
        self.resnetBlock512_2 = ResnetConvBlock(512, 512, use_batchnorm=use_batchnorm)
        self.resnetBlock512_3 = ResnetConvBlock(512, 512, use_batchnorm=use_batchnorm)

        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.inputConv(x)
        x = self.pool1(x)

        x = self.resnetBlock64_1(x)
        x = self.resnetBlock64_2(x)
        x = self.resnetBlock64_3(x)

        x = self.resnetBlock128_1(x)
        x = self.resnetBlock128_2(x)
        x = self.resnetBlock128_3(x)
        x = self.resnetBlock128_4(x)

        x = self.resnetBlock256_1(x)
        x = self.resnetBlock256_2(x)
        x = self.resnetBlock256_3(x)
        x = self.resnetBlock256_4(x)
        x = self.resnetBlock256_5(x)
        x = self.resnetBlock256_6(x)

        x = self.resnetBlock512_1(x)
        x = self.resnetBlock512_2(x)
        x = self.resnetBlock512_3(x)

        x = self.avgPool(x)
        x = torch.flatten(x, 1)

        return self.head(x)

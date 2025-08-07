"From github user @milesil (https://github.com/milesial/Pytorch-UNet)"

import torch
import torch.nn as nn

from math import floor


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2. Adapted to use leaky ReLu"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        k: int = 3,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=k,
                padding=floor(k / 2),
                bias=True,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=k,
                padding=floor(k / 2),
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        k: int = 3,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.up: nn.Sequential | nn.ConvTranspose2d

        self.conv = DoubleConv(in_channels, out_channels, in_channels, k=k, padding_mode=padding_mode)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                k,
                padding=floor(k / 2),
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.up(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 3,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        # could also try just changing the stride the conv
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, k=k, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

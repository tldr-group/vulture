import torch
from torch import compile
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.benchmark as benchmark
from utils import measure_mem_time

torch.backends.cudnn.benchmark = False


from math import log2, floor, ceil


# From github user @milesil (https://github.com/milesial/Pytorch-UNet)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2. Adapted to use leaky ReLu"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        learned: bool = True,
    ):
        super().__init__()
        self.up: nn.Upsample | nn.ConvTranspose2d
        # if bilinear, use the normal convolutions to reduce the number of channels
        if learned:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class Downsampler(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_in: int = 3,
        n_ch_out: int = 16,
        learned: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_ch_out = n_ch_out

        n_downsamples = ceil(log2(patch_size))
        prev_ch = n_ch_in

        layers = []
        for i in range(n_downsamples):
            out_ch = n_ch_out
            downsample: Down | nn.MaxPool2d
            if learned:
                downsample = Down(prev_ch, out_ch)
            else:
                downsample = nn.MaxPool2d(2)
            layers.append(downsample)
            prev_ch = out_ch
        self.downsample_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        _, _, H, W = x.shape
        out_h, out_w = floor(H / self.patch_size), floor(W / self.patch_size)
        downsamples: list[torch.Tensor] = []
        x_prev = x
        for layer in self.downsample_layers:
            downsamples.append(x_prev)
            x_prev = layer(x_prev)
        downsamples.append(F.interpolate(x_prev, (out_h, out_w)))
        return downsamples


class Upsampler(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_in: int = 128,
        n_ch_downsample: int = 16,
        learned: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size

        upsamples: list[nn.Module] = []
        n_upsamples = ceil(log2(patch_size))
        for i in range(n_upsamples):
            upsample = Up(n_ch_in + n_ch_downsample, n_ch_in, learned)
            upsamples.append(upsample)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(
        self, lr_feats: torch.Tensor, downsamples: list[torch.Tensor]
    ) -> torch.Tensor:
        x = lr_feats
        _, _, double_h, double_w = downsamples[1].shape
        _, _, out_h, out_w = downsamples[-1].shape

        i = 0
        for layer, guidance in zip(self.upsamples, downsamples):
            if i == 1:
                x = F.interpolate(x, (double_h, double_w))
            x_in = torch.cat((x, guidance), dim=1)
            x = layer(x_in)
            i += 1
        x = F.interpolate(x, (out_h, out_w))
        return x


class Combined(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_img: int = 3,
        n_ch_in: int = 128,
        n_ch_downsample: int = 16,
        learned: bool = True,
    ):
        super().__init__()

        self.downsampler = Downsampler(patch_size, n_ch_img, n_ch_downsample, learned)
        self.upsampler = Upsampler(patch_size, n_ch_in, n_ch_downsample, learned)

    def forward(self, img: torch.Tensor, lr_feats: torch.Tensor) -> torch.Tensor:
        downsamples: list[torch.Tensor] = self.downsampler(img)[::-1]
        return self.upsampler(lr_feats, downsamples)


def test_benchmark():
    l = 1400
    test_down_str = f"""
                from __main__ import Downsampler
                d = Downsampler(14, 3, 16, True).to('cuda:0')
                test= torch.ones((1, 3, {l}, {l}), device='cuda:0')
                """

    test_both_str = f"""
                from __main__ import Downsampler, Upsampler
                d = Downsampler(14, 3, 16, True).to('cuda:0')
                u = Upsampler(14, 128, 16).to('cuda:0')
                test= torch.ones((1, 3, {l}, {l}), device='cuda:0')
                test_lr = torch.ones((1, 128, {l//14}, {l//14}), device='cuda:0')
                """
    t0 = benchmark.Timer(
        stmt="d(test)",
        setup=test_down_str,
    )
    print(t0.timeit(10))

    t1 = benchmark.Timer(
        stmt="downs = d(test)\nu(test_lr, downs[::-1])",
        setup=test_both_str,
    )
    print(t1.timeit(10))


if __name__ == "__main__":
    combined = Combined(14).to("cuda:1")

    l = 224
    test = torch.ones((20, 3, l, l), device="cuda:1")
    test_lr = torch.ones((20, 128, l // 14, l // 14), device="cuda:1")

    mem, time = measure_mem_time(test, test_lr, combined)

    print(f"{mem}MB, {time}s")

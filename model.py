import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.benchmark as benchmark
from utils import measure_mem_time

torch.backends.cudnn.benchmark = False

import numpy as np
from math import log2, floor, ceil


# from featup
class ImplicitFeaturizer(torch.nn.Module):

    def __init__(
        self,
        color_feats: bool = True,
        n_freqs: int = 10,
        learn_bias: bool = False,
        time_feats: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.color_feats = color_feats
        self.time_feats = time_feats
        self.n_freqs = n_freqs
        self.learn_bias = learn_bias

        self.dim_multiplier = 2

        if self.color_feats:
            self.dim_multiplier += 3

        if self.time_feats:
            self.dim_multiplier += 1

        if self.learn_bias:
            self.biases = torch.nn.Parameter(
                torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32)
            )

    def forward(self, original_image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat(
            [t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]
        ).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(
            torch.linspace(-2, 10, self.n_freqs, device=original_image.device)
        ).reshape(1, self.n_freqs, 1, 1, 1)
        feats = feats * freqs

        if self.learn_bias:
            sin_feats = feats + self.biases[0].reshape(
                1, self.n_freqs, self.dim_multiplier, 1, 1
            )
            cos_feats = feats + self.biases[1].reshape(
                1, self.n_freqs, self.dim_multiplier, 1, 1
            )
        else:
            sin_feats = feats
            cos_feats = feats

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        return torch.cat(all_feats, dim=1)


# From github user @milesil (https://github.com/milesial/Pytorch-UNet)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2. Adapted to use leaky ReLu"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        k: int = 3,
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
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=k,
                padding=floor(k / 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, k: int = 3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, k=k)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        k: int = 3,
        learned: bool = True,
    ):
        super().__init__()
        self.up: nn.Upsample | nn.ConvTranspose2d
        # if bilinear, use the normal convolutions to reduce the number of channels
        if learned:
            self.conv = DoubleConv(in_channels, in_channels, k=k)
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, k=k)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.up(x)


class Downsampler(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_in: int = 3,
        n_ch_out: int = 16,
        k: int | list[int] = 3,
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
                current_k = k if isinstance(k, int) else k[i]
                downsample = Down(prev_ch, out_ch, current_k)
            else:
                downsample = nn.MaxPool2d(2)
            layers.append(downsample)
            prev_ch = out_ch
        self.downsample_layers = nn.ModuleList(layers)

        self.n_freqs = 10
        self.implict = ImplicitFeaturizer(True, self.n_freqs)
        self.freq_dims = 3 + self.n_freqs * 10

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        _, _, H, W = x.shape
        out_h, out_w = floor(H / self.patch_size), floor(W / self.patch_size)
        downsamples: list[torch.Tensor] = []
        x_prev, x_impl = x, self.implict(x)
        x_cat = torch.cat((x_prev, x_impl), dim=1)
        for layer in self.downsample_layers:
            downsamples.append(x_cat)
            x_prev = layer(x_prev)
            _, _, h, w = x_prev.shape
            x_impl = self.implict(F.interpolate(x, (h, w)))
            x_cat = torch.cat((x_prev, x_impl), dim=1)

        x_final = F.interpolate(x_prev, (out_h, out_w))
        x_impl = self.implict(F.interpolate(x, (out_h, out_w)))
        downsamples.append(torch.cat((x_final, x_impl), dim=1))
        return downsamples


class Upsampler(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_in: int = 128,
        n_ch_downsample: int = 119,
        k: int | list[int] = 3,
        learned: bool = True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.inp_conv = DoubleConv(n_ch_in, n_ch_in, None, 5)
        self.outp_conv = nn.Conv2d(n_ch_in, n_ch_in, 3, padding=1)
        self.mlp = nn.Conv2d(n_ch_in, n_ch_in, 1)
        # consider add a conv kernel size=1 at the end and removing tanh activation
        self.act = nn.Tanh()

        upsamples: list[nn.Module] = []
        n_upsamples = ceil(log2(patch_size))
        for i in range(n_upsamples):
            current_k = k if isinstance(k, int) else k[i]
            upsample = Up(n_ch_in + n_ch_downsample, n_ch_in, current_k, learned)
            upsamples.append(upsample)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(
        self, lr_feats: torch.Tensor, downsamples: list[torch.Tensor]
    ) -> torch.Tensor:
        x = lr_feats
        _, _, double_h, double_w = downsamples[1].shape
        _, _, out_h, out_w = downsamples[-1].shape

        x = self.inp_conv(x)  # mix info before Conv2DT
        i = 0
        for layer, guidance in zip(self.upsamples, downsamples):
            if i == 1:
                x = F.interpolate(x, (double_h, double_w))
            x_in = torch.cat((x, guidance), dim=1)
            x = layer(x_in)
            i += 1
        x = self.outp_conv(x)  # conv w/out LReLu activation
        # x = self.act(x)
        x = self.mlp(x)
        x = F.interpolate(x, (out_h, out_w))
        return x


class Combined(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_img: int = 3,
        n_ch_in: int = 128,
        n_ch_downsample: int = 16,
        k_down: int | list[int] = 3,
        k_up: int | list[int] = 3,
        learned: bool = True,
    ):
        super().__init__()

        self.downsampler = Downsampler(
            patch_size, n_ch_img, n_ch_downsample, k_down, learned
        )
        n_guidance_dims = n_ch_downsample + self.downsampler.freq_dims
        self.upsampler = Upsampler(patch_size, n_ch_in, n_guidance_dims, k_up, learned)

    def forward(self, img: torch.Tensor, lr_feats: torch.Tensor) -> torch.Tensor:
        downsamples: list[torch.Tensor] = self.downsampler(img)[::-1]
        return self.upsampler(lr_feats, downsamples)


class Simple(nn.Module):
    def __init__(
        self,
        n_ch_out: int = 128,
        n_convs: int = 8,
        k: int = 3,
        contractive: bool = True,
    ):
        super().__init__()
        n_freqs = 10
        n_guidance_dims = 3  # + n_freqs * 10
        self.implict = ImplicitFeaturizer(True, n_freqs)

        if contractive:
            ch_vals = np.linspace(
                n_guidance_dims + n_ch_out,
                n_ch_out,
                n_convs + 1,
                endpoint=True,
                dtype=np.int32,
            )
            ch_vals = [int(i) for i in ch_vals]
        else:
            ch_vals = [n_ch_out for i in range(n_convs + 1)]
        print(ch_vals)
        self.in_conv = nn.Conv2d(
            n_guidance_dims + n_ch_out, ch_vals[0], k, padding=floor(k / 2)
        )
        layers: list[nn.Module] = []
        for i in range(1, n_convs + 1):
            layers.append(DoubleConv(ch_vals[i - 1], ch_vals[i], k=k))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, lr_feats: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        resized_feats = F.interpolate(lr_feats, (h, w))
        # impl_feats = self.implict(x)
        inp = torch.cat((x, resized_feats), dim=1)
        x = self.in_conv(inp)
        for l in self.layers:
            x = l(x)
        x = F.normalize(x, p=1, dim=1)
        return x


# TODO:
# - convs before conv2DT


def test_benchmark():
    l = 1400
    test_down_str = f"""
                from __main__ import Downsampler
                d = Downsampler(14).to('cuda:0')
                test= torch.ones((1, 3, {l}, {l}), device='cuda:0')
                """

    test_both_str = f"""
                from __main__ import Downsampler, Upsampler
                d = Downsampler(14).to('cuda:0')
                u = Upsampler(14).to('cuda:0')
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
    # torch.cuda.empty_cache()
    combined = Simple(n_convs=8).to("cuda:0").eval()

    l = 400
    test = torch.ones((1, 3, l, l), device="cuda:0")
    test_lr = torch.ones((1, 128, l // 14, l // 14), device="cuda:0")

    out: torch.Tensor = combined(test, test_lr)
    print(out.shape)

    mem, time = measure_mem_time(test, test_lr, combined)

    print(f"{mem}MB, {time}s")

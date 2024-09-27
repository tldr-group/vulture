import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ImplicitFeaturizer, DoubleConv, Up, Down

import torch.utils.benchmark as benchmark
from utils import measure_mem_time

torch.backends.cudnn.benchmark = False

import numpy as np
from math import log2, floor, ceil


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


class Skips(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_freq_impl: int = 10,
        n_ch_in: int = 128,
        k_up: int | list[int] = 5,
        lr_weight: float = 0.4,
        learned: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.lr_weight = lr_weight
        self.implict = ImplicitFeaturizer(True, n_freq_impl, True)
        n_ch_downsample = 3 + n_freq_impl * 10

        upsamples: list[nn.Module] = []
        self.n_upsamples = ceil(log2(patch_size))
        for i in range(self.n_upsamples):
            current_k = k_up if isinstance(k_up, int) else k_up[i]
            upsample = Up(n_ch_in + n_ch_downsample, n_ch_in, current_k, learned)
            upsamples.append(upsample)
        self.upsamples = nn.ModuleList(upsamples)

        # post_upsample_convs: list[nn.Module] = []
        # for i in range(3):
        #     post_upsample_convs.append(
        #         DoubleConv(n_ch_in + n_ch_downsample, n_ch_in, current_k)
        #     )
        # self.post_upsample_convs = nn.ModuleList(post_upsample_convs)
        self.inp_conv = DoubleConv(n_ch_in, n_ch_in, None, 5)
        self.conv = DoubleConv(n_ch_in + n_ch_downsample, n_ch_in, k=current_k)
        self.outp_conv = nn.Conv2d(n_ch_in, n_ch_in, 3, padding=1)
        self.mlp = nn.Conv2d(n_ch_in, n_ch_in, 1)

    def _get_sizes(self, h: int, w: int, n_upsamples: int) -> list[tuple[int, int]]:
        sizes: list[tuple[int, int]] = []
        for i in range(n_upsamples + 1):
            sizes.append((int(h / 2**i), int(w / 2**i)))
        return sizes

    def forward(self, img: torch.Tensor, lr_feats: torch.Tensor):
        _, _, H, W = img.shape
        impl = self.implict(img)

        sizes = self._get_sizes(H, W, self.n_upsamples)[::-1]
        x_prev = F.interpolate(lr_feats, sizes[0])
        for s, size in enumerate(sizes):
            x_prev = F.interpolate(x_prev, size)
            guidance = F.interpolate(impl, size)
            resized_lr_feats = F.interpolate(lr_feats, size)

            lr_weight = self.lr_weight / (2 * (s + 1))
            x_prev = ((1 - lr_weight) * x_prev) * (lr_weight * resized_lr_feats)
            # x_prev = x_prev * resized_lr_feats
            x_cat = torch.cat((x_prev, guidance), dim=1)
            if s < len(sizes) - 1:
                x_prev = self.upsamples[s](x_cat)
            else:
                x_prev = self.conv(x_cat)

        # for i in range(1, len(self.post_upsample_convs)):
        #     x_prev = torch.cat((x_prev, guidance), dim=1)
        #     x_prev = self.post_upsample_convs[i](x_prev)

        x = self.outp_conv(x_prev)
        x = self.mlp(x)
        x = F.normalize(x, 1, dim=1)
        return x


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
    torch.cuda.empty_cache()
    l = 1400
    test = torch.ones((1, 3, l, l), device="cuda:0")
    test_lr = torch.ones((1, 128, l // 14, l // 14), device="cuda:0")
    print(test_lr.shape)

    net = Skips(14, 10).to("cuda:0").eval()
    x = net.forward(test, test_lr)
    print(x.shape)

    mem, time = measure_mem_time(test, test_lr, net)

    print(f"{mem}MB, {time}s")

    """
    combined = Simple(n_convs=8).to("cuda:0").eval()

    l = 400
    test = torch.ones((1, 3, l, l), device="cuda:0")
    test_lr = torch.ones((1, 128, l // 14, l // 14), device="cuda:0")

    out: torch.Tensor = combined(test, test_lr)
    print(out.shape)

    mem, time = measure_mem_time(test, test_lr, combined)

    print(f"{mem}MB, {time}s")
    """

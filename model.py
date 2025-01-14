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
        n_ch_out: int = 64,
        n_ch_guidance: int = 3,
        k: int | list[int] = 3,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_ch_out = n_ch_out

        n_downsamples = ceil(log2(patch_size))
        layers = []
        for i in range(n_downsamples):
            out_ch = n_ch_out
            prev_ch = n_ch_out + n_ch_guidance if i > 0 else n_ch_in
            current_k = k if isinstance(k, int) else k[i]

            downsample = Down(prev_ch, out_ch, current_k, padding_mode=padding_mode)
            layers.append(downsample)
        self.downsample_layers = nn.ModuleList(layers)

        # self.n_freqs = 10
        # self.implict = ImplicitFeaturizer(True, self.n_freqs)
        # self.freq_dims = 3 + self.n_freqs * 10

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        _, _, H, W = img.shape
        out_h, out_w = floor(H / self.patch_size), floor(W / self.patch_size)
        downsamples: list[torch.Tensor] = []

        x_cat = img
        for layer in self.downsample_layers:
            downsamples.append(x_cat)
            x_prev = layer(x_cat)
            _, _, h, w = x_prev.shape
            resized_guidance = F.interpolate(img, (h, w))
            x_cat = torch.cat((x_prev, resized_guidance), dim=1)

        x_final = F.interpolate(x_cat, (out_h, out_w))
        downsamples.append(x_final)
        return downsamples


class Upsampler(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_in: int = 128,
        n_ch_out: int = 128,
        n_ch_downsample: int = 67,
        k: int | list[int] = 3,
        add_feats: bool = False,
        feat_weight: float = 0.3,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.inp_conv = DoubleConv(n_ch_in, n_ch_in, None, 5, padding_mode=padding_mode)
        self.outp_conv = nn.Conv2d(
            n_ch_out, n_ch_out, 3, padding=1, padding_mode=padding_mode
        )
        self.mlp = nn.Conv2d(n_ch_out, n_ch_out, 1)

        upsamples: list[nn.Module] = []
        self.n_upsamples = ceil(log2(patch_size))

        if n_ch_in != n_ch_out:
            chs = list(np.linspace(n_ch_in, n_ch_out, self.n_upsamples, dtype=np.int32))
        else:
            chs = [n_ch_in for i in range(self.n_upsamples)]

        for i in range(self.n_upsamples):
            current_k = k if isinstance(k, int) else k[i]
            in_ch = n_ch_in if i == 0 else chs[i - 1]
            upsample = Up(
                in_ch + n_ch_downsample, chs[i], current_k, padding_mode=padding_mode
            )
            upsamples.append(upsample)
        self.upsamples = nn.ModuleList(upsamples)

        self.add_lr_guidance = add_feats
        self.lr_weight = feat_weight

    def forward(
        self, lr_feats: torch.Tensor, downsamples: list[torch.Tensor]
    ) -> torch.Tensor:
        x = lr_feats
        _, _, double_h, double_w = downsamples[1].shape
        _, _, out_h, out_w = downsamples[-1].shape

        x = self.inp_conv(x)  # mix info before Conv2DT
        i = 0
        for layer, guidance in zip(self.upsamples, downsamples):
            _, _, gh, gw = guidance.shape
            x = F.interpolate(x, (gh, gw))
            # if i == 1:
            #    x = F.interpolate(x, (double_h, double_w))
            if (i < self.n_upsamples // 2) and self.add_lr_guidance:
                _, _, h, w = x.shape
                resized_lr_feats = F.interpolate(lr_feats, (h, w))
                x = ((1 - self.lr_weight) * x) * (self.lr_weight * resized_lr_feats)
            # print(x.shape, guidance.shape)
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
        n_ch_out: int = 128,
        n_ch_downsample: int = 64,
        k_down: int | list[int] = 3,
        k_up: int | list[int] = 3,
        feat_weight: float = -1,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.downsampler = Downsampler(
            patch_size, n_ch_img, n_ch_downsample, k=k_down, padding_mode=padding_mode
        )
        n_guidance_dims = n_ch_downsample + 3  # + self.downsampler.freq_dims
        add_feats = feat_weight > 0
        self.upsampler = Upsampler(
            patch_size,
            n_ch_in,
            n_ch_out,
            n_guidance_dims,
            k_up,
            add_feats=add_feats,
            feat_weight=feat_weight,
            padding_mode=padding_mode,
        )

    def forward(self, img: torch.Tensor, lr_feats: torch.Tensor) -> torch.Tensor:
        downsamples: list[torch.Tensor] = self.downsampler(img)[::-1]
        return self.upsampler(lr_feats, downsamples)


class SimpleConv(nn.Module):
    def __init__(
        self,
        n_ch_out: int = 128,
        n_ch_guidance: int = 64,
        n_convs: int = 8,
        k: int = 3,
        contractive: bool = True,
    ):
        super().__init__()

        if contractive:
            ch_vals = np.linspace(
                n_ch_guidance + n_ch_out,
                n_ch_out,
                n_convs + 1,
                endpoint=True,
                dtype=np.int32,
            )
            ch_vals = [int(i) for i in ch_vals]
        else:
            ch_vals = [n_ch_out for i in range(n_convs + 1)]

        self.in_conv = nn.Conv2d(
            n_ch_guidance + n_ch_out, ch_vals[0], k, padding=floor(k / 2)
        )
        layers: list[nn.Module] = []
        for i in range(1, n_convs + 1):
            layers.append(DoubleConv(ch_vals[i - 1], ch_vals[i], k=k))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for l in self.layers:
            x = l(x)
        # x = F.normalize(x, p=1, dim=1)
        return x


class FeaturePropagator(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_imgs: int = 6,
        n_ch_in: int = 128,
        n_ch_out: int = 128,
        n_ch_downsample: int = 64,
        k: int | list[int] = 3,
        k_down: int | list[int] = 3,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.downsampler = Downsampler(
            patch_size, n_ch_imgs, n_ch_downsample, k=k_down, padding_mode=padding_mode
        )
        self.


# class Skips(nn.Module):
#     def __init__(
#         self,
#         patch_size: int,
#         n_freq_impl: int = 10,
#         n_ch_in: int = 128,
#         k_up: int | list[int] = 5,
#         lr_weight: float = 0.4,
#         learned: bool = False,
#     ):
#         super().__init__()
#         self.patch_size = patch_size

#         self.lr_weight = lr_weight
#         self.implict = ImplicitFeaturizer(True, n_freq_impl, True)
#         n_ch_downsample = 3 + n_freq_impl * 10

#         upsamples: list[nn.Module] = []
#         self.n_upsamples = ceil(log2(patch_size))
#         for i in range(self.n_upsamples):
#             current_k = k_up if isinstance(k_up, int) else k_up[i]
#             upsample = Up(n_ch_in + n_ch_downsample, n_ch_in, current_k, learned)
#             upsamples.append(upsample)
#         self.upsamples = nn.ModuleList(upsamples)

#         # post_upsample_convs: list[nn.Module] = []
#         # for i in range(3):
#         #     post_upsample_convs.append(
#         #         DoubleConv(n_ch_in + n_ch_downsample, n_ch_in, current_k)
#         #     )
#         # self.post_upsample_convs = nn.ModuleList(post_upsample_convs)
#         self.inp_conv = DoubleConv(n_ch_in, n_ch_in, None, 5)
#         self.conv = DoubleConv(n_ch_in + n_ch_downsample, n_ch_in, k=current_k)
#         self.outp_conv = nn.Conv2d(n_ch_in, n_ch_in, 3, padding=1)
#         self.mlp = nn.Conv2d(n_ch_in, n_ch_in, 1)

#     def _get_sizes(self, h: int, w: int, n_upsamples: int) -> list[tuple[int, int]]:
#         sizes: list[tuple[int, int]] = []
#         for i in range(n_upsamples + 1):
#             sizes.append((int(h / 2**i), int(w / 2**i)))
#         return sizes

#     def forward(self, img: torch.Tensor, lr_feats: torch.Tensor):
#         _, _, H, W = img.shape
#         impl = self.implict(img)

#         sizes = self._get_sizes(H, W, self.n_upsamples)[::-1]
#         x_prev = F.interpolate(lr_feats, sizes[0])
#         for s, size in enumerate(sizes):
#             x_prev = F.interpolate(x_prev, size)
#             guidance = F.interpolate(impl, size)
#             resized_lr_feats = F.interpolate(lr_feats, size)

#             if s < len(sizes) // 2:
#                 lr_weight = self.lr_weight / (2 * (s + 1))
#                 x_prev = ((1 - lr_weight) * x_prev) * (lr_weight * resized_lr_feats)
#             # x_prev = x_prev * resized_lr_feats
#             x_cat = torch.cat((x_prev, guidance), dim=1)
#             if s < len(sizes) - 1:
#                 x_prev = self.upsamples[s](x_cat)
#             else:
#                 x_prev = self.conv(x_cat)

#         # for i in range(1, len(self.post_upsample_convs)):
#         #     x_prev = torch.cat((x_prev, guidance), dim=1)
#         #     x_prev = self.post_upsample_convs[i](x_prev)

#         x = self.outp_conv(x_prev)
#         x = self.mlp(x)
#         x = F.normalize(x, 1, dim=1)
#         return x


# class FeatureTransfer(nn.Module):
#     def __init__(
#         self,
#         n_ch_img: int = 3,
#         n_ch_in: int = 384,
#         n_ch_out: int = 128,
#         k: int = 5,
#         depth: int = 3,
#         padding_mode: str = "zeros",
#     ):
#         super().__init__()
#         self.n_ch_img = n_ch_img
#         if n_ch_in != n_ch_out:
#             chs = list(np.linspace(n_ch_in, n_ch_out, depth, dtype=np.int32))
#         else:
#             chs = [n_ch_in for i in range(depth)]

#         convs: list[nn.Module] = []
#         for d in range(depth):
#             in_ch = n_ch_in if d == 0 else chs[d - 1]
#             conv = DoubleConv(in_ch + n_ch_img, chs[d], k=k, padding_mode=padding_mode)
#             convs.append(conv)
#         self.convs = nn.ModuleList(convs)
#         self.mlp = nn.Conv2d(chs[-1] + n_ch_img, chs[-1], 1)

#     def forward(self, img: torch.Tensor, lr_feats: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = lr_feats.shape
#         img_lr = F.interpolate(img, (h, w))
#         x = torch.cat((lr_feats, img_lr), dim=1) if self.n_ch_img > 0 else lr_feats
#         for layer in self.convs:
#             x = layer(x)
#             x = torch.cat((x, img_lr), dim=1) if self.n_ch_img > 0 else x
#         x = self.mlp(x)
#         return x


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
    l = 224 + 14
    test = torch.ones((1, 3, 2 * l, l), device="cuda:0")
    test_lr = torch.ones((1, 384, (2 * l) // 14, l // 14), device="cuda:0")

    net = (
        Combined(
            14,
            n_ch_in=384,
        )
        .to("cuda:0")
        .eval()
    )
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

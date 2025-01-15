import torch
import torch.nn as nn
import torch.nn.functional as F

from yoeo.models.layers import DoubleConv, Up, Down

import torch.utils.benchmark as benchmark
from yoeo.utils import measure_mem_time

torch.backends.cudnn.benchmark = False

import numpy as np
from math import log2, floor, ceil


class LearnedDownsampler(nn.Module):
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

        self.add_feat_guidance = add_feats
        self.lr_weight = feat_weight

    def forward(
        self, lr_feats: torch.Tensor, downsamples: list[torch.Tensor]
    ) -> torch.Tensor:
        x = lr_feats
        _, _, out_h, out_w = downsamples[-1].shape

        x = self.inp_conv(x)  # mix info before Conv2DT
        i = 0
        for layer, guidance in zip(self.upsamples, downsamples):
            _, _, gh, gw = guidance.shape
            x = F.interpolate(x, (gh, gw))

            # if we want to re-add original features to early layers - not recommended
            if (i < self.n_upsamples // 2) and self.add_feat_guidance:
                _, _, h, w = x.shape
                resized_lr_feats = F.interpolate(lr_feats, (h, w))
                x = ((1 - self.lr_weight) * x) * (self.lr_weight * resized_lr_feats)

            x_in = torch.cat((x, guidance), dim=1)
            x = layer(x_in)
            i += 1
        x = self.outp_conv(x)  # conv w/out LReLu activation
        # x = self.act(x)
        x = self.mlp(x)
        x = F.interpolate(x, (out_h, out_w))
        return x


class FeatureUpsampler(nn.Module):
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

        self.downsampler = LearnedDownsampler(
            patch_size, n_ch_img, n_ch_downsample, k=k_down, padding_mode=padding_mode
        )
        n_guidance_dims = n_ch_downsample + 3
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
        n_ch_in: int = 128,
        n_ch_out: int = 128,
        n_ch_guidance: int = 64,
        n_convs: int = 8,
        k: int = 3,
        contractive: bool = True,
    ):
        super().__init__()

        ch_vals: list[int] | np.ndarray
        if contractive:
            ch_vals = np.linspace(
                n_ch_guidance + n_ch_in,
                n_ch_out,
                n_convs + 1,
                endpoint=True,
                dtype=np.int32,
            )
            ch_vals = [int(i) for i in ch_vals]
        else:
            ch_vals = [n_ch_out for i in range(n_convs + 1)]

        self.in_conv = nn.Conv2d(
            n_ch_guidance + n_ch_in, ch_vals[0], k, padding=floor(k / 2)
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
        n_layers: int = 5,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.downsampler = LearnedDownsampler(
            patch_size, n_ch_imgs, n_ch_downsample, k=k_down, padding_mode=padding_mode
        )
        self.feature_adjuster = SimpleConv(
            n_ch_in, n_ch_out, n_ch_downsample, n_layers, k, True
        )

    def forward(
        self, f0: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor
    ) -> torch.Tensor:
        combined_imgs = torch.cat((i0, i1), dim=1)  # (B, 2C, H, W)
        guidance_from_image_features = self.downsampler.forward(combined_imgs)[0]
        input_feats = torch.cat((f0, guidance_from_image_features), dim=1)
        f1 = self.feature_adjuster.forward(input_feats)
        return f1


def test_benchmark():
    l = 1400
    test_down_str = f"""
                from __main__ import LearnedDownsampler
                d = Downsampler(14).to('cuda:0')
                test= torch.ones((1, 3, {l}, {l}), device='cuda:0')
                """

    test_both_str = f"""
                from __main__ import LearnedDownsampler, Upsampler
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
        FeatureUpsampler(
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

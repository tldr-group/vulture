import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import numpy as np
from math import log2, floor, ceil

from yoeo.models.layers import DoubleConv, Up, Down
from yoeo.utils import measure_mem_time, get_n_params

from typing import Literal
from dataclasses import dataclass

PaddingModes = Literal["zeros", "reflect", "replicate", "circular"]
InitTypes = Literal["ones", "zeros", "xavier", "uniform", "default"]


@dataclass
class UpsamplerConfig:
    name: str
    # conv kernel size
    k: int = 3
    # number of input channels: 384 for full Dv2, 128/64/32 for Featup PCA'd inputs
    n_ch_in: int = 128
    # number of hidden channels: -1 means same as input
    n_ch_hidden: int = -1
    # number of output channels: usually same as n_ch)in
    n_ch_out: int = 128
    # number of 'guidance' channels (i.e channel dim of image)
    n_ch_guidance: int = 3
    # number of hidden dims in downsampler (usally 64 or 32)
    n_ch_downsampler: int = 64
    # if resizing and adding in original features during upsampling - not used
    feat_weight: float = -1
    # vit model patch size (usually 14)
    patch_size: int = 14
    padding_mode: PaddingModes = "zeros"
    # Weights init
    weights_init: InitTypes = "default"


def init_weights(m: nn.Module, init: InitTypes):
    if isinstance(m, nn.Linear):
        if init == "ones":
            torch.nn.init.constant_(m.weight, 1)
        elif init == "zeros":
            torch.nn.init.constant_(m.weight, 0)
        elif init == "xavier":
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.01)
        else:
            pass


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
        n_ch_deep: int = -1,
        n_ch_out: int = 128,
        n_ch_downsample: int = 67,
        k: int | list[int] = 3,
        add_feats: bool = False,
        feat_weight: float = 0.3,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.patch_size = patch_size
        n_ch_deep = n_ch_deep if n_ch_deep > -1 else n_ch_in  # if defined
        self.inp_conv = DoubleConv(n_ch_in, n_ch_deep, None, 5, padding_mode=padding_mode)

        upsamples: list[nn.Module] = []
        self.n_upsamples = ceil(log2(patch_size))

        if n_ch_in != n_ch_out:
            chs = list(np.linspace(n_ch_deep, n_ch_out, self.n_upsamples, dtype=np.int32))
        else:
            chs = [n_ch_deep for i in range(self.n_upsamples)]

        for i in range(self.n_upsamples):
            current_k = k if isinstance(k, int) else k[i]
            in_ch = n_ch_deep if i == 0 else chs[i - 1]
            upsample = Up(in_ch + n_ch_downsample, chs[i], current_k, padding_mode=padding_mode)
            upsamples.append(upsample)
        self.upsamples = nn.ModuleList(upsamples)

        self.outp_conv = nn.Conv2d(chs[-1], n_ch_out, 3, padding=1, padding_mode=padding_mode)
        self.mlp = nn.Conv2d(n_ch_out, n_ch_out, 1)

        self.add_feat_guidance = add_feats
        self.lr_weight = feat_weight

    def forward(self, lr_feats: torch.Tensor, downsamples: list[torch.Tensor]) -> torch.Tensor:
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
        x = self.mlp(x)
        x = F.interpolate(x, (out_h, out_w))
        return x


class FeatureUpsampler(nn.Module):
    def __init__(
        self,
        patch_size: int,
        n_ch_img: int = 3,
        n_ch_in: int = 128,
        n_ch_deep: int = -1,
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
        n_guidance_dims = n_ch_downsample + n_ch_img
        add_feats = feat_weight > 0
        self.upsampler = Upsampler(
            patch_size,
            n_ch_in,
            n_ch_deep,
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


def get_upsampler(
    chk_path: str | None,
    upsampler_cfg: UpsamplerConfig | None = None,
    device: str = "cpu",
    to_eval: bool = False,
    to_half: bool = False,
) -> FeatureUpsampler:
    if upsampler_cfg is not None and chk_path is None:
        cfg = upsampler_cfg
        model = FeatureUpsampler(
            cfg.patch_size,
            n_ch_img=cfg.n_ch_guidance,
            n_ch_in=cfg.n_ch_in,
            n_ch_deep=cfg.n_ch_hidden,
            n_ch_out=cfg.n_ch_out,
            n_ch_downsample=cfg.n_ch_downsampler,
            k_up=cfg.k,
            feat_weight=cfg.feat_weight,
            padding_mode=cfg.padding_mode,
        )
    elif chk_path is not None:
        obj = torch.load(chk_path, weights_only=True, map_location=device)
        try:
            cfg = UpsamplerConfig(**obj["config"])
        except KeyError:
            assert upsampler_cfg is not None, f"No config found in {chk_path}, one must be supplied!"
            cfg = upsampler_cfg

        model = FeatureUpsampler(
            cfg.patch_size,
            n_ch_img=cfg.n_ch_guidance,
            n_ch_in=cfg.n_ch_in,
            n_ch_deep=cfg.n_ch_hidden,
            n_ch_out=cfg.n_ch_out,
            n_ch_downsample=cfg.n_ch_downsampler,
            k_up=cfg.k,
            feat_weight=cfg.feat_weight,
            padding_mode=cfg.padding_mode,
        )
        model.load_state_dict(obj["weights"])
    else:
        raise Exception("One of chk_path or autoencoder_cfg must be supplied")

    model = model.to(device)
    if to_eval:
        model = model.eval()
    if to_half:
        model = model.half()
    return model


def test_benchmark() -> None:
    image_l = 1400
    test_down_str = f"""
                from __main__ import LearnedDownsampler
                d = Downsampler(14).to('cuda:0')
                test= torch.ones((1, 3, {image_l}, {image_l}), device='cuda:0')
                """

    test_both_str = f"""
                from __main__ import LearnedDownsampler, Upsampler
                d = Downsampler(14).to('cuda:0')
                u = Upsampler(14).to('cuda:0')
                test= torch.ones((1, 3, {image_l}, {image_l}), device='cuda:0')
                test_lr = torch.ones((1, 128, {image_l // 14}, {image_l // 14}), device='cuda:0')
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
    image_l = 1000
    test = torch.ones((1, 3, 2 * image_l, image_l), device="cuda:0")
    test_lr = torch.ones((1, 384, (2 * image_l) // 14, image_l // 14), device="cuda:0")

    net = FeatureUpsampler(14, n_ch_in=384, n_ch_deep=96, n_ch_out=384, n_ch_downsample=32).to("cuda:0").eval()
    print(f"N_params: {get_n_params(net)}")
    with torch.no_grad():
        x = net.forward(test, test_lr)
        print(x.shape)

        mem, time = measure_mem_time(test, test_lr, net)

    print(f"{mem}MB, {time}s")

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor


# From github user @milesil (https://github.com/milesial/Pytorch-UNet)
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
        """
        if learned:
            self.conv = DoubleConv(in_channels, in_channels, k=k)
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
        """
        self.conv = DoubleConv(
            in_channels, out_channels, in_channels, k=k, padding_mode=padding_mode
        )
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


# from featup
# class ImplicitFeaturizer(nn.Module):

#     def __init__(
#         self,
#         color_feats: bool = True,
#         n_freqs: int = 10,
#         learn_bias: bool = False,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.color_feats = color_feats
#         self.n_freqs = n_freqs
#         self.learn_bias = learn_bias

#         self.dim_multiplier = 2

#         if self.color_feats:
#             self.dim_multiplier += 3

#         if self.learn_bias:
#             self.biases = torch.nn.Parameter(
#                 torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32)
#             )

#     def forward(self, original_image: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = original_image.shape
#         grid_h = torch.linspace(-1, 1, h, device=original_image.device)
#         grid_w = torch.linspace(-1, 1, w, device=original_image.device)
#         feats = torch.cat(
#             [t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]
#         ).unsqueeze(0)
#         feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

#         if self.color_feats:
#             feat_list = [feats, original_image]
#         else:
#             feat_list = [feats]

#         feats = torch.cat(feat_list, dim=1).unsqueeze(1)
#         freqs = torch.exp(
#             torch.linspace(-2, 10, self.n_freqs, device=original_image.device)
#         ).reshape(1, self.n_freqs, 1, 1, 1)
#         feats = feats * freqs

#         if self.learn_bias:
#             sin_feats = feats + self.biases[0].reshape(
#                 1, self.n_freqs, self.dim_multiplier, 1, 1
#             )
#             cos_feats = feats + self.biases[1].reshape(
#                 1, self.n_freqs, self.dim_multiplier, 1, 1
#             )
#         else:
#             sin_feats = feats
#             cos_feats = feats

#         sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)
#         cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

#         if self.color_feats:
#             all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
#         else:
#             all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

#         return torch.cat(all_feats, dim=1)

import torch
import torch.nn as nn

from typing import Literal
from dataclasses import dataclass

Optims = Literal["Adam", "AdamW", "SGD"]
Losses = Literal["MSE", "MAE"]


@dataclass
class Config:
    name: str
    in_ch: int
    base_ch: int
    n_layers: int
    k: int
    optim: Optims
    loss_type: Losses
    lr: float
    batch_size: int
    n_epochs: int


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        n_layers: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        layers: list[torch.nn.Module] = []
        channels = in_channels
        for i in range(n_layers):
            out_channels = max(1, base_channels // (2**i))
            layers.append(
                nn.Conv2d(
                    channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.LeakyReLU(inplace=True))
            channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        n_layers: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        layers: list[torch.nn.Module] = []
        channels = max(1, base_channels // (2 ** (n_layers - 1)))
        for i in range(n_layers - 1, 0, -1):
            next_channels = max(1, base_channels // (2 ** (i - 1)))
            layers.append(
                nn.Conv2d(
                    channels,
                    next_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.LeakyReLU(inplace=True))
            channels = next_channels
        layers.append(
            nn.Conv2d(
                channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        n_layers: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, n_layers, kernel_size, stride)
        self.decoder = Decoder(in_channels, base_channels, n_layers, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def get_autoencoder(chk_path: str, device: str) -> Autoencoder:
    obj = torch.load(chk_path, weights_only=True)
    cfg = Config(**obj["config"])
    model = Autoencoder(cfg.in_ch, cfg.base_ch, cfg.n_layers, cfg.k)
    model.load_state_dict(obj["weights"])
    model = model.to(device)
    return model


if __name__ == "__main__":
    D = 384
    e = Encoder(D, D, 4, 1)
    d = Decoder(D, D, 4, 1)

    x = torch.ones((1, D, 14, 14))
    z = e(x)
    x_pred = d(z)
    print(z.shape, x_pred.shape)

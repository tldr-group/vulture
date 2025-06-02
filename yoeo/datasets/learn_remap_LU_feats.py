from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from PIL import Image
from os import listdir
import numpy as np

from yoeo.utils import do_2D_pca
import matplotlib.pyplot as plt


from typing import Mapping, Any

torch.random.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda:0"


def get_train_data(lr_feats: torch.Tensor, hr_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, c, n_t_h, n_t_w = lr_feats.shape
    # _, c, h, w = hr_feats.shape

    # lr_hr_feats = F.interpolate(hr_feats, (n_t_h, n_t_w), mode="bilinear")
    lr_hr_feats = F.avg_pool2d(hr_feats, (14, 14), 14)

    lr_flat: torch.Tensor = lr_feats.reshape((c, -1)).permute((1, 0))
    lr_hr_flat: torch.Tensor = lr_hr_feats.reshape((c, -1)).permute((1, 0))

    return (lr_hr_flat, lr_flat)


# hr_lr_feats = F.interpolate(lr_feats, (h, w), mode="bilinear")

# lr_flat: torch.Tensor = hr_lr_feats.reshape((c, -1)).permute((1, 0))
# lr_hr_flat: torch.Tensor = hr_feats.reshape((c, -1)).permute((1, 0))

# return (lr_hr_flat, lr_flat)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),  # Output dimension matches input
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_mlp(
    train_data: torch.Tensor,
    target_data: torch.Tensor,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose=False,
    n_dims: int = 384,
    device: str = "cuda:0",
) -> MLP:
    model = MLP(n_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = TensorDataset(train_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    return model


@torch.no_grad()
def apply(mlp: MLP, hr_feats: torch.Tensor) -> torch.Tensor:
    _, c, h, w = hr_feats.shape
    hr_flat = hr_feats.reshape((c, -1)).permute((1, 0))
    res = mlp.forward(hr_flat)
    res_2D = res.reshape((1, h, w, c)).permute((0, 3, 1, 2))
    return res_2D


def vis(
    save_path: str,
    img: Image.Image | None,
    lr_feats: torch.Tensor,
    hr_feats: torch.Tensor,
    weights: Mapping[str, Any] | None,
    mlp: MLP | None = None,
):
    _, c, h_, _ = hr_feats.shape

    if mlp is None:
        assert weights
        mlp = MLP(c, 384)
        mlp.load_state_dict(weights)
        mlp = mlp.to(hr_feats.device)

    res_2D = apply(mlp, hr_feats)

    lr_feats_np = lr_feats.cpu()[0].numpy()
    hr_feats_np = hr_feats.cpu()[0].numpy()
    res_2D_np = res_2D.detach().cpu()[0].numpy()

    lr_feats_red = do_2D_pca(lr_feats_np, 3, post_norm="minmax")
    hr_feats_red = do_2D_pca(hr_feats_np, 3, post_norm="minmax")
    res_feats_red = do_2D_pca(res_2D_np, 3, post_norm="minmax")

    fig, axs = plt.subplots(ncols=4)
    axs[0].imshow(img)
    axs[1].imshow(lr_feats_red)
    axs[2].imshow(hr_feats_red)
    axs[3].imshow(res_feats_red)

    for ax in axs:
        ax.set_axis_off()

    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    PATH = "data/imagenet_reduced"
    DATA_FOLDER = "data_lu_reg"

    fname = "00245"

    img = Image.open(f"{PATH}/imgs/{fname}.png")
    data = torch.load(f"{PATH}/{DATA_FOLDER}/{fname}.pt")
    lr_feats: torch.Tensor = data["lr_feats"]
    hr_feats: torch.Tensor = data["hr_feats"]

    train, targ = get_train_data(lr_feats, hr_feats)

    mlp = train_mlp(train, targ, 32, 1000, lr=1e-3, verbose=True)

    res = apply(mlp, hr_feats)

    vis("tmp/better_remap.png", img, lr_feats, hr_feats, None, mlp)


# if __name__ == "__main__":
#     PATH = "data/imagenet_reduced"
#     DATA_FOLDER = "data_lu_reg"
#     fnames = sorted(listdir(f"{PATH}/{DATA_FOLDER}"))
#     N = len(fnames)

#     for i, fname in enumerate(fnames):
#         img = Image.open(f"{PATH}/imgs/{fname.split('.')[0]}.png")
#         data = torch.load(f"{PATH}/{DATA_FOLDER}/{fname}", weights_only=True, map_location=DEVICE)

#         lr_feats: torch.Tensor = data["lr_feats"]
#         hr_feats: torch.Tensor = data["hr_feats"]

#         train, targ = get_train_data(lr_feats, hr_feats)
#         mlp = train_mlp(train, targ, batch_size=512, epochs=1000, lr=1e-3)

#         data["mlp_weights"] = mlp.state_dict()
#         torch.save(data, f"{PATH}/{DATA_FOLDER}/{fname}")

#         if i % 10 == 0:
#             print(f"[{i:03d} / {N}]")


# print(f"{end - start}s ")
# print(mlp)

# print(res.shape)
# torch.save(mlp, "tmp/mlp.pt")
# VIS

# res = mlp.forward(hr_flat)
# res_2D = res.reshape((1, h, w, c)).permute((0, 3, 1, 2))


# lr_feats_np = lr_feats.cpu()[0].numpy()
# hr_feats_np = hr_feats.cpu()[0].numpy()
# res_2D_np = res_2D.detach().cpu()[0].numpy()

# lr_feats_red = do_2D_pca(lr_feats_np, 3, post_norm="minmax")
# hr_feats_red = do_2D_pca(hr_feats_np, 3, post_norm="minmax")
# res_feats_red = do_2D_pca(res_2D_np, 3, post_norm="minmax")


# fig, axs = plt.subplots(ncols=4)
# axs[0].imshow(img)
# axs[1].imshow(lr_feats_red)
# axs[2].imshow(hr_feats_red)
# axs[3].imshow(res_feats_red)

# for ax in axs:
#     ax.set_axis_off()

# plt.savefig("tmp/compare_pca_2_layers.png", bbox_inches="tight")

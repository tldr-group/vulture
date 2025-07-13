import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from PIL import Image

from yoeo.comparisons.loftup.featurizers.util import get_featurizer
from yoeo.comparisons.online_denoiser import Denoiser
from yoeo.comparisons.autoencoder import get_autoencoder
from yoeo.utils import do_2D_pca
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train(
    train_data: torch.Tensor,
    target_data: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose=False,
    n_dims: int = 384,
    device: str = "cuda:0",
) -> Model:
    _, c, _, _ = train_data.shape
    model = Model(c, n_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        output = model(train_data)
        output_lr = F.avg_pool2d(output, (14, 14), 14)
        loss = criterion(output_lr, target_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.5f}")

    return model


@torch.no_grad()
def apply(mlp: Model, hr_feats: torch.Tensor) -> torch.Tensor:
    res_2D = mlp.forward(hr_feats)
    return res_2D


def rescale(a: np.ndarray) -> np.ndarray:
    a_min = a.min(axis=(0, 1), keepdims=True)
    a_max = a.max(axis=(0, 1), keepdims=True)
    out = (a - a_min) / (a_max - a_min)
    return out


def vis(
    save_path: str,
    img: Image.Image | None,
    lr_feats: torch.Tensor,
    hr_feats: torch.Tensor,
    remapped: torch.Tensor | None,
    save_hr_separate_as_well: bool = False,
    is_featup: bool = False,
):
    lr_feats_np = lr_feats.cpu()[0].numpy().astype(np.float32)
    hr_feats_np = hr_feats.cpu()[0].numpy().astype(np.float32)

    if is_featup:
        lr_feats_red = lr_feats_np.transpose((1, 2, 0))[:, :, 0:3]
        lr_feats_red = rescale(lr_feats_red)
        hr_feats_red = hr_feats_np.transpose((1, 2, 0))[:, :, 0:3]
        hr_feats_red = rescale(hr_feats_red)
    else:
        lr_feats_red = do_2D_pca(lr_feats_np, 3, post_norm="minmax")
        hr_feats_red = do_2D_pca(hr_feats_np, 3, post_norm="minmax")

    n_cols = 4 if remapped is not None else 3

    _, axs = plt.subplots(ncols=n_cols, figsize=(n_cols * 10, 10))
    axs[0].imshow(img)
    axs[1].imshow(lr_feats_red)
    axs[2].imshow(hr_feats_red)
    if remapped is not None:
        res_2D_np = remapped.detach().cpu()[0].numpy()
        res_feats_red = do_2D_pca(res_2D_np, 3, post_norm="minmax")
        axs[3].imshow(res_feats_red)

    for ax in axs:
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

    if save_hr_separate_as_well:
        sep_path = save_path.split(".")[0] + "_hr.png"
        plt.imsave(sep_path, hr_feats_red)


if __name__ == "__main__":
    PATH = "data/imagenet_reduced"
    DATA_FOLDER = "data_lu_reg"

    for i in range(0, 30):
        fname = f"00{240 + i}"

        DEVICE = "cuda:1"

        featurizer_class = "dinov2s_reg"
        torch_hub_name = "loftup_dinov2s_reg"

        model, patch_size, dim = get_featurizer(featurizer_class)
        model = model.to(DEVICE)

        kernel_size = patch_size
        lr_size = 224 // patch_size  # 2 * 224 // patch_size
        load_size = 224

        upsampler = torch.hub.load("andrehuang/loftup", torch_hub_name, pretrained=True)
        upsampler = upsampler.to(DEVICE).eval()

        denoiser = Denoiser(feat_dim=384).to(DEVICE)
        denoiser_weights = torch.load("yoeo/comparisons/vit_small_patch14_reg4_dinov2.lvd142m.pth")
        denoiser.load_state_dict(denoiser_weights["denoiser"])

        autoencoder = get_autoencoder("trained_models/dac_dv2_denoised_e500.pth", DEVICE)

        img = Image.open(f"{PATH}/imgs/{fname}.png")
        # data = torch.load(f"{PATH}/{DATA_FOLDER}/{fname}.pt")

        MU, SIGMA = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        to_norm_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=MU, std=SIGMA),
            ]
        )

        transform = T.Compose(
            [
                T.Resize(load_size, T.InterpolationMode.BILINEAR),
                T.CenterCrop(load_size),  # Depending on whether you want a center crop
                to_norm_tensor,
            ]
        )

        normalized_img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lr_feats = model(normalized_img_tensor)
            lr_feats = lr_feats.permute((0, 2, 3, 1))
            lr_feats = denoiser.forward(lr_feats, return_channel_first=True)
            hr_feats = upsampler(lr_feats, normalized_img_tensor)  # 1, dim, 224, 224
            # lr_feats: torch.Tensor = data["lr_feats"].to(DEVICE)
            # hr_feats: torch.Tensor = data["hr_feats"].to(DEVICE)
            print(lr_feats.shape, hr_feats.shape)

            compressed_lr = autoencoder.encoder(F.normalize(lr_feats, p=1, dim=1))
            compressed_hr = autoencoder.encoder(F.normalize(hr_feats, p=1, dim=1))

        mlp = train(compressed_hr, compressed_lr, 3000, lr=1e-3, verbose=True, device=DEVICE, n_dims=48)

        res = apply(mlp, compressed_hr)
        # print(torch.sum(mlp.model.weight.data))
        print(res.shape)

        vis(f"tmp/remap_2/{i}.png", img, compressed_lr, hr_feats, res)

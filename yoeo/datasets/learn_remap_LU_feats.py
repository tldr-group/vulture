import torch
from torch import nn
import torch.nn.functional as F

from PIL import Image

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
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    return model


@torch.no_grad()
def apply(mlp: Model, hr_feats: torch.Tensor) -> torch.Tensor:
    res_2D = mlp.forward(hr_feats)
    return res_2D


def vis(
    save_path: str, img: Image.Image | None, lr_feats: torch.Tensor, hr_feats: torch.Tensor, remapped: torch.Tensor
):
    lr_feats_np = lr_feats.cpu()[0].numpy()
    hr_feats_np = hr_feats.cpu()[0].numpy()
    res_2D_np = remapped.detach().cpu()[0].numpy()

    lr_feats_red = do_2D_pca(lr_feats_np, 3, post_norm="minmax")
    hr_feats_red = do_2D_pca(hr_feats_np, 3, post_norm="minmax")
    res_feats_red = do_2D_pca(res_2D_np, 3, post_norm="minmax")

    _, axs = plt.subplots(ncols=4)
    axs[0].imshow(img)
    axs[1].imshow(lr_feats_red)
    axs[2].imshow(hr_feats_red)
    axs[3].imshow(res_feats_red)

    for ax in axs:
        ax.set_axis_off()

    plt.savefig(save_path, bbox_inches="tight")


# if __name__ == "__main__":
#     PATH = "data/imagenet_reduced"
#     DATA_FOLDER = "data_lu_reg"

#     fname = "00235"

#     DEVICE = "cuda:1"

#     featurizer_class = "dinov2s_reg"
#     torch_hub_name = "loftup_dinov2s_reg"

#     model, patch_size, dim = get_featurizer(featurizer_class)
#     model = model.to(DEVICE)

#     kernel_size = patch_size
#     lr_size = 224 // patch_size  # 2 * 224 // patch_size
#     load_size = 224

#     upsampler = torch.hub.load("andrehuang/loftup", torch_hub_name, pretrained=True)
#     upsampler = upsampler.to(DEVICE).eval()

#     denoiser = Denoiser(feat_dim=384).to(DEVICE)
#     denoiser_weights = torch.load("yoeo/comparisons/vit_small_patch14_reg4_dinov2.lvd142m.pth")
#     denoiser.load_state_dict(denoiser_weights["denoiser"])

#     img = Image.open(f"{PATH}/imgs/{fname}.png")
#     data = torch.load(f"{PATH}/{DATA_FOLDER}/{fname}.pt")

#     transform = T.Compose(
#         [
#             T.Resize(load_size, T.InterpolationMode.BILINEAR),
#             T.CenterCrop(load_size),  # Depending on whether you want a center crop
#             T.ToTensor(),
#             norm,
#         ]
#     )

#     normalized_img_tensor = transform(img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         lr_feats = model(normalized_img_tensor)
#         # lr_feats = lr_feats.permute((0, 2, 3, 1))
#         # lr_feats = denoiser.forward(lr_feats, return_channel_first=True)
#         hr_feats = upsampler(lr_feats, normalized_img_tensor)  # 1, dim, 224, 224
#     # lr_feats: torch.Tensor = data["lr_feats"].to(DEVICE)
#     # hr_feats: torch.Tensor = data["hr_feats"].to(DEVICE)

#     mlp = train(hr_feats, lr_feats, 2000, lr=1e-3, verbose=True, device=DEVICE)

#     res = apply(mlp, hr_feats)
#     # print(torch.sum(mlp.model.weight.data))
#     print(res.shape)

#     vis("tmp/better_better_remap.png", img, lr_feats, hr_feats, res)

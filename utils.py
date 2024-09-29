import torch
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func

from timm.models.vision_transformer import VisionTransformer, Attention, Block  # type: ignore
from types import MethodType
from typing import Callable, Literal

from time import time_ns
from PIL import Image
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

NormType = Literal["minmax", "std", None]
norm_dict = {
    "minmax": MinMaxScaler(feature_range=(0, 1), clip=True, copy=False),
    "std": StandardScaler(copy=False),
}


# ========================= UPGRADES =========================
class Patch:
    @staticmethod
    def add_flash_attn() -> Callable:
        """Replaces normal 'forward()' method of the memory efficient attention layer (block.attn)
        in the Dv2 model with an optional early return with attention. Used if xformers used.

        :return: the new forward method
        :rtype: Callable
        """

        def forward(
            self: Attention,
            x: torch.Tensor,
            attn_bias=None,
        ) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            x = flash_attn_qkvpacked_func(qkv)  # type: ignore
            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        return forward


def add_flash_attention(model: VisionTransformer) -> VisionTransformer:
    blk: Block
    for blk in model.blocks:  # type: ignore
        blk.attn.forward = MethodType(Patch.add_flash_attn(), blk.attn)
    return model


# ========================= PCA STUFF =========================
def flatten(x: torch.Tensor | np.ndarray) -> np.ndarray:
    y: np.ndarray
    if type(x) == torch.Tensor:
        y = to_numpy(x)
    else:
        y = x  # type: ignore
    c, h, w = y.shape
    y = y.reshape((c, h * w))
    y = y.T
    return y


def do_pca(
    arr: np.ndarray,
    n_components: int = 3,
    n_samples: int = -1,
    pre_norm: NormType = None,
    post_norm: NormType = None,
) -> np.ndarray:
    # arr in shape (n_samples, n_features)
    if n_samples > -1:
        inds = np.arange(arr.shape[0])
        sample_inds = np.random.choice(inds, n_samples)
        train_data = arr[sample_inds]
    else:
        train_data = arr

    if pre_norm != None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[pre_norm]
        scaler.fit_transform(arr)
        # arr = scaler.transform(arr)
        # train_data = scaler.transform(train_data)

    pca = PCA(n_components=n_components)

    train_proj = pca.fit_transform(train_data)
    projection = pca.transform(arr)

    if post_norm != None:
        scaler: MinMaxScaler | StandardScaler = norm_dict[post_norm]
        scaler.fit_transform(projection)
        # projection = scaler.transform(projection)
    return projection


def do_2D_pca(
    arr_2D: np.ndarray,
    n_components: int = 3,
    n_samples: int = -1,
    pre_norm: NormType = None,
    post_norm: NormType = None,
) -> np.ndarray:
    c, h, w = arr_2D.shape
    flat = flatten(arr_2D)
    proj = do_pca(flat, n_components, n_samples, pre_norm, post_norm)
    proj_2d = proj.reshape((h, w, n_components))
    return proj_2d


# ========================= INPUT TRANSFORMS =========================

MU, SIGMA = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
INV_MU = (-MU[0] / SIGMA[0], -MU[1] / SIGMA[1], -MU[2] / SIGMA[2])
INV_SIGMA = (1 / SIGMA[0], 1 / SIGMA[1], 1 / SIGMA[2])

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

to_norm_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=MU, std=SIGMA),
    ]
)

unnormalize = transforms.Normalize(
    mean=INV_MU,
    std=INV_SIGMA,
)


def closest_crop(
    h: int, w: int, patch_size: int = 14, to_tensor: bool = True
) -> transforms.Compose:
    # Crop to h,w values that are closest to given patch/stride size
    sub_h: int = h % patch_size
    sub_w: int = w % patch_size
    new_h, new_w = h - sub_h, w - sub_w
    if to_tensor:
        transform = transforms.Compose(
            [transforms.CenterCrop((new_h, new_w)), to_norm_tensor]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.CenterCrop((new_h, new_w)),
            ]
        )
    return transform


def load_image(
    path: str,
    transform: transforms.Compose,
    to_gpu: bool = True,
    to_half: bool = True,
    batch: bool = True,
) -> tuple[torch.Tensor, Image.Image]:
    # Load image with PIL, convert to tensor by applying $transform, and invert transform to get display image
    image = Image.open(path).convert("RGB")
    tensor: torch.Tensor = convert_image(image, transform, to_gpu, to_half, batch)
    transformed_img = to_img(unnormalize(tensor.squeeze(0)))
    return tensor, transformed_img


def convert_image(
    img: Image.Image,
    transform: transforms.Compose,
    to_gpu: bool = True,
    to_half: bool = True,
    batch: bool = True,
) -> torch.Tensor:
    tensor: torch.Tensor = transform(img)  # type: ignore
    if to_half:
        tensor = tensor.to(torch.float16)
    if to_gpu:
        tensor = tensor.to("cuda:0")
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor


# ========================= PERF =========================


def measure_mem_time(
    img: torch.Tensor,
    feats: torch.Tensor,
    model: torch.nn.Module,
) -> tuple[float, float]:

    torch.cuda.reset_peak_memory_stats(feats.device)  # s.t memory is accurate
    torch.cuda.synchronize(feats.device)  # s.t time is accurate

    def _to_MB(x: int) -> float:
        return x / (1024**2)

    def _to_s(t: int) -> float:
        return t / 1e9

    start_m = torch.cuda.max_memory_allocated(feats.device)
    start_t = time_ns()

    model.forward(img, feats)

    end_m = torch.cuda.max_memory_allocated(feats.device)
    torch.cuda.synchronize(feats.device)
    end_t = time_ns()

    return _to_MB(end_m - start_m), _to_s(end_t - start_t)


# ========================= VISUALISE =========================


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if len(arr.shape) == 4:
        arr = arr[0]
    return arr


def get_arrs_from_batch(
    img: torch.Tensor,
    lr_feats: torch.Tensor,
    hr_feats: torch.Tensor,
    pred_hr_feats: torch.Tensor | None,
) -> list[list[np.ndarray]]:
    b, c, h, w = hr_feats.shape

    arrs: list[list[np.ndarray]] = []
    for i in range(b):
        img_tensor, lr_feat_tensor, hr_feat_tensor, pred_hr_tensor = (
            img[i],
            lr_feats[i],
            hr_feats[i],
            pred_hr_feats[i],
        )
        img_arr = to_numpy(img_tensor.permute((1, 2, 0)))

        out_2D_arrs: list[np.ndarray] = [img_arr]
        tensors = (
            (lr_feat_tensor, hr_feat_tensor, pred_hr_tensor)
            if isinstance(pred_hr_feats, torch.Tensor)
            else (lr_feat_tensor, hr_feat_tensor)
        )
        for i, d in enumerate(tensors):
            feat_arr = to_numpy(d)
            k = 3
            pca = PCA(n_components=k)

            n_c, h, w = feat_arr.shape
            data_flat = feat_arr.reshape((n_c, h * w)).T
            out = pca.fit_transform(data_flat)
            out_rescaled = MinMaxScaler().fit_transform(out)

            out_2D = out_rescaled.reshape((h, w, k))
            out_2D_arrs.append(out_2D)
        arrs.append(out_2D_arrs)
    return arrs


# put vis code in here
def visualise(
    img: torch.Tensor | Image.Image,
    lr_feats: torch.Tensor,
    hr_feats: torch.Tensor,
    pred_hr_feats: torch.Tensor | None,
    out_path: str,
) -> None:
    # b, c, h, w = hr_feats.shape
    n_rows = 4 if isinstance(pred_hr_feats, torch.Tensor) else 3
    arrs = get_arrs_from_batch(img, lr_feats, hr_feats, pred_hr_feats)
    fig, axs = plt.subplots(nrows=n_rows, ncols=len(arrs))
    fig.set_size_inches(32, 4.4)
    for i, arr in enumerate(arrs):
        for j, sub_arr in enumerate(arr):
            if len(arrs) == 1:
                axs[j].imshow(sub_arr)
                axs[j].set_axis_off()
            else:
                axs[j, i].imshow(sub_arr)
                axs[j, i].set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_losses(train_loss: list[float], val_loss: list[float], out_path: str) -> None:
    epochs = np.arange(len(train_loss))
    plt.semilogy(epochs, train_loss, lw=2, label="train")
    plt.semilogy(epochs, val_loss, lw=2, label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func

from timm.models.vision_transformer import VisionTransformer, Attention, Block  # type: ignore

import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from time import time_ns
from PIL import Image
import matplotlib.pyplot as plt

from types import MethodType
from typing import Callable, Literal
from collections import defaultdict
from dataclasses import dataclass, field

from json import load as load_json

from tqdm import tqdm
import torch.autograd.profiler as profiler


import warnings

warnings.filterwarnings("ignore")

NormType = Literal["minmax", "std", None]
norm_dict = {
    "minmax": MinMaxScaler(feature_range=(0, 1), clip=True, copy=False),
    "std": StandardScaler(copy=False),
}
InitTypes = Literal["ones", "zeros", "xavier", "uniform", "default"]

# ========================= TYPES =========================


@dataclass
class Experiment:
    name: str
    net_type: Literal["combined", "simple", "skips", "transfer"] = "combined"
    k: int = 3
    n_ch_in: int = 384
    n_ch_out: int = 128
    n_ch_guidance: int = 3
    n_ch_downsampler: int = 64
    feat_weight: float = -1
    patch_size: int = 14
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"

    flip_h_prob: float = 0.5
    flip_v_prob: float = 0.5
    angles_deg: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 90, 180, 270])
    shift_dirs: list[tuple[int, int]] = field(default_factory=lambda: [])
    shift_dists: list[int] = field(default_factory=lambda: [])
    norm: bool = True

    loss: Literal["smooth_l1", "l1", "l2"] = "smooth_l1"
    optim: Literal["adamw", "adam", "SGD"] = "adamw"
    lr: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 5000
    save_per: int = 10
    weights_init: InitTypes = "default"


def expriment_from_json(json_obj_or_path: dict | str) -> Experiment:
    config: dict
    if isinstance(json_obj_or_path, str):
        with open(json_obj_or_path) as f:
            config = load_json(f)
    else:
        config = json_obj_or_path
    for key in ("net", "transforms", "training"):
        for subkey, subval in config[key].items():
            config[subkey] = subval
        config.pop(key)
    config["net_type"] = config["type"]
    config.pop("type")

    return Experiment(**config)


# ========================= UPGRADES =========================


def init_weights(m: nn.Module, init: InitTypes):
    if isinstance(m, nn.Linear):
        if init == "unit":
            torch.nn.init.constant_(m.weight, 1)
        elif init == "zero":
            torch.nn.init.constant_(m.weight, 0)
        elif init == "xavier":
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.01)
        else:
            pass


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
    # projection = pca.transform(arr)
    projection = arr[:, :3]

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


def resize_crop(
    resize_dims: tuple[int, int], crop_dims: tuple[int, int]
) -> transforms.Compose:
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dims),
            transforms.CenterCrop(crop_dims),
            to_norm_tensor,
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
        tensor = tensor.to("cuda:1")
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


def paired_frames_vis(
    imgs_0: torch.Tensor, imgs_1: torch.Tensor, out_path: str
) -> None:
    def _torch_to_np(x: torch.Tensor):
        return unnormalize(x).numpy().transpose((0, 2, 3, 1)).astype(np.uint8)

    vis_0, vis_1 = (
        _torch_to_np(imgs_0),
        _torch_to_np(imgs_1),
    )
    N = vis_1.shape[0]
    fig, axs = plt.subplots(nrows=2, ncols=N)
    fig.set_size_inches(4 * N, 4 * 2)

    for i in range(N):
        axs[0, i].imshow(vis_0[i])
        axs[0, i].set_axis_off()
        axs[1, i].imshow(vis_1[i])
        axs[1, i].set_axis_off()

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


# ========================= FEATUP SILLINESS =========================
def apply_jitter(img, max_pad, transform_params):
    h, w = img.shape[2:]

    padded = F.pad(img, [max_pad] * 4, mode="reflect")

    zoom = transform_params["zoom"].item()
    x = transform_params["x"].item()
    y = transform_params["y"].item()
    flip = transform_params["flip"].item()

    if zoom > 1.0:
        zoomed = F.interpolate(padded, scale_factor=zoom, mode="bilinear")
    else:
        zoomed = padded

    cropped = zoomed[:, :, x : h + x, y : w + y]

    if flip:
        return torch.flip(cropped, [3])
    else:
        return cropped


def sample_transform(use_flips, max_pad, max_zoom, h, w):
    if use_flips:
        flip = random.random() > 0.5
    else:
        flip = False

    apply_zoom = random.random() > 0.5
    if apply_zoom:
        zoom = random.random() * (max_zoom - 1) + 1
    else:
        zoom = 1.0

    valid_area_h = (int((h + max_pad * 2) * zoom) - h) + 1
    valid_area_w = (int((w + max_pad * 2) * zoom) - w) + 1

    return {
        "x": torch.tensor(torch.randint(0, valid_area_h, ()).item()),
        "y": torch.tensor(torch.randint(0, valid_area_w, ()).item()),
        "zoom": torch.tensor(zoom),
        "flip": torch.tensor(flip),
    }


class JitteredImage(Dataset):

    def __init__(self, img, length, use_flips, max_zoom, max_pad):
        self.img = img
        self.length = length
        self.use_flips = use_flips
        self.max_zoom = max_zoom
        self.max_pad = max_pad

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        h, w = self.img.shape[2:]
        transform_params = sample_transform(
            self.use_flips, self.max_pad, self.max_zoom, h, w
        )
        return (
            apply_jitter(self.img, self.max_pad, transform_params).squeeze(0),
            transform_params,
        )


def get_lr_feats(
    model, img: torch.Tensor, n_imgs: int = 50, fit3d: bool = False
) -> torch.Tensor:
    cfg_n_images = n_imgs  # 3000  # 3000
    cfg_use_flips = True
    cfg_max_zoom = 1.8
    cfg_max_pad = 30
    cfg_pca_batch = 50
    cfg_proj_dim = 128

    def project(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = img.shape

        n_patch_w: int = 1 + (w - 14) // 14
        n_patch_h: int = 1 + (h - 14) // 14
        if fit3d:
            feats = model.forward_features(x)[:, 5:, :]
        else:
            feats = model.forward_features(x)["x_norm_patchtokens"]
        b, _, c = feats.shape

        return feats.permute((0, 2, 1)).reshape((b, c, n_patch_h, n_patch_w))

    dataset = JitteredImage(img, cfg_n_images, cfg_use_flips, cfg_max_zoom, cfg_max_pad)
    loader = DataLoader(dataset, cfg_pca_batch)
    with torch.no_grad():
        lr_feats = project(img)

        jit_features = []
        for transformed_image, tp in loader:
            jit_features.append(project(transformed_image))
        jit_features = torch.cat(jit_features, dim=0)
        # transform_params = {k: torch.cat(v, dim=0) for k, v in transform_params.items()}

        unprojector = PCAUnprojector(
            jit_features[:cfg_pca_batch],
            cfg_proj_dim,
            lr_feats.device,
            use_torch_pca=True,
        )
        # jit_features = unprojector.project(jit_features)
        lr_feats = unprojector.project(lr_feats)
    return lr_feats


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return (
            tensor.permute(1, 0, 2, 3)
            .reshape(C, B * H * W)
            .permute(1, 0)
            # .detach()
            # .cpu()
        )

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x.to(torch.float32))
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(
            x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2)
        )  # .to(device)

    return reduced_feats, fit_pca


class PCAUnprojector(nn.Module):

    def __init__(self, feats, dim, device, use_torch_pca=False, **kwargs):
        super().__init__()
        self.dim = dim

        if feats is not None:
            self.original_dim = feats.shape[1]
        else:
            self.original_dim = kwargs["original_dim"]

        if self.dim != self.original_dim:
            if feats is not None:
                sklearn_pca = pca([feats], dim=dim, use_torch_pca=use_torch_pca)[1]

                # Register tensors as buffers
                self.register_buffer(
                    "components_",
                    torch.tensor(
                        sklearn_pca.components_, device=device, dtype=feats.dtype
                    ),
                )
                self.register_buffer(
                    "singular_values_",
                    torch.tensor(
                        sklearn_pca.singular_values_, device=device, dtype=feats.dtype
                    ),
                )
                self.register_buffer(
                    "mean_",
                    torch.tensor(sklearn_pca.mean_, device=device, dtype=feats.dtype),
                )
            else:
                self.register_buffer("components_", kwargs["components_"].t())
                self.register_buffer("singular_values_", kwargs["singular_values_"])
                self.register_buffer("mean_", kwargs["mean_"])

        else:
            print("PCAUnprojector will not transform data")

    def forward(self, red_feats):
        if self.dim == self.original_dim:
            return red_feats
        else:
            b, c, h, w = red_feats.shape
            red_feats_reshaped = red_feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            # print(red_feats_reshaped.dtype, self.components_.dtype)
            red_feats_reshaped = red_feats_reshaped.to(self.components_.dtype)
            unprojected = (
                red_feats_reshaped @ self.components_
            ) + self.mean_.unsqueeze(0)
            return unprojected.reshape(b, h, w, self.original_dim).permute(0, 3, 1, 2)

    def project(self, feats):
        if self.dim == self.original_dim:
            return feats
        else:
            b, c, h, w = feats.shape
            feats_reshaped = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            t0 = feats_reshaped - self.mean_.unsqueeze(0).to(feats.device)
            projected = t0 @ self.components_.t().to(feats.device)
            return projected.reshape(b, h, w, self.dim).permute(0, 3, 1, 2)


def prep_image(t, subtract_min=True):
    if subtract_min:
        t -= t.min()
    t /= t.max()
    t = (t * 255).clamp(0, 255).to(torch.uint8)

    if len(t.shape) == 2:
        t = t.unsqueeze(0)

    return t


if __name__ == "__main__":
    DEVICE = "cuda:1"
    dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dv2 = add_flash_attention(dv2)
    dv2 = dv2.eval().to(DEVICE).half()
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        img = torch.zeros((1, 3, 518, 518)).half().to(DEVICE)
        res = get_lr_feats(dv2, img, 25)

    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_time_total", row_limit=5
        )
    )

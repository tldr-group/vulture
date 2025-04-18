import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

import numpy as np

from yoeo.models.model import FeatureUpsampler
from yoeo.feature_prep import get_lr_feats, PCAUnprojector
from yoeo.utils import (
    add_flash_attention,
    expriment_from_json,
    closest_crop,
    convert_image,
    Experiment,
)


def get_dv2_model(
    fit_3d: bool = True,
    add_flash: bool = True,
    to_half: bool = True,
    device: str | torch.device = "cuda:0",
) -> torch.nn.Module:
    if fit_3d:
        dv2 = torch.hub.load("ywyue/FiT3D", "dinov2_reg_small_fine")
    else:
        dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

    if add_flash:
        dv2 = add_flash_attention(dv2)

    if to_half:
        dv2 = dv2.half()

    dv2 = dv2.eval().to(device)
    return dv2


def get_upsampler_and_expr(
    chk_path: str,
    cfg_path: str,
    device: str | torch.device = "cuda:0",
) -> tuple[FeatureUpsampler, Experiment]:
    upsampler_weights = torch.load(chk_path, weights_only=True, map_location=device)

    expr = expriment_from_json(cfg_path)

    upsampler = FeatureUpsampler(
        expr.patch_size,
        n_ch_img=expr.n_ch_guidance,
        n_ch_in=expr.n_ch_in,
        n_ch_out=expr.n_ch_out,
        n_ch_downsample=expr.n_ch_downsampler,
        k_up=expr.k,
        feat_weight=expr.feat_weight,
        padding_mode=expr.padding_mode,
    )
    upsampler.load_state_dict(upsampler_weights)
    upsampler = upsampler.eval().to(device)
    return upsampler, expr


@torch.no_grad()
def get_hr_feats(
    image: Image.Image | np.ndarray,
    dv2: torch.nn.Module,
    upsampler: FeatureUpsampler,
    device: str | torch.device = "cuda:0",
    fit_3d: bool = True,
    n_imgs_for_red: int = 50,
    n_ch_in: int = 64,
    n_batch_lr: int = 50,
    existing_pca: PCAUnprojector | None = None
):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    _h, _w = image.height, image.width
    tr = closest_crop(_h, _w)
    cropped_tensor = convert_image(image, tr, batch=True, device_str=device)

    inp_img = (
        TF.normalize(
            TF.pil_to_tensor(image).to(torch.float32),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        .unsqueeze(0)
        .to(device)
    )
    reduced_tensor, _ = get_lr_feats(
        dv2,
        [cropped_tensor.half()],
        n_imgs_for_red,
        fit3d=fit_3d,
        n_feats_in=n_ch_in,
        n_batch=n_batch_lr,
        existing_pca=existing_pca
    )
    reduced_tensor = F.normalize(reduced_tensor, p=1, dim=1)

    hr_feats: torch.Tensor
    with torch.autocast("cuda", torch.float16):
        hr_feats = upsampler(inp_img, reduced_tensor)
    return hr_feats

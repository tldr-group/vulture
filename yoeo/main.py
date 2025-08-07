import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

import numpy as np

from yoeo.models.external.vit_wrapper import add_flash_attention
from yoeo.models import (
    PretrainedViTWrapper,
    Denoiser,
    Autoencoder,
    FeatureUpsampler,
    AutoencoderConfig,
    UpsamplerConfig,
    get_denoiser,
    get_autoencoder,
    get_upsampler,
    MODEL_MAP,
    FeatureType,
)
from yoeo.feature_prep import get_lr_feats, PCAUnprojector, get_lr_featup_feats_and_pca
from yoeo.utils import (
    expriment_from_json,
    closest_crop,
    convert_image,
    Experiment,
)


def transform_image(
    image: np.ndarray | Image.Image | torch.Tensor, device: str, to_half: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    if isinstance(image, torch.Tensor):
        _h, _w = image.shape[-2:]
        batch = len(image.shape) < 4
    else:
        _h, _w = image.height, image.width
        batch = True

    tr = closest_crop(_h, _w)
    lr_feat_input_img = convert_image(
        image, tr, batch=batch, to_gpu=device != "cpu", device_str=device, to_half=to_half
    )

    upsampler_input_img = (
        TF.normalize(
            TF.pil_to_tensor(image).to(torch.float32),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        .unsqueeze(0)
        .to(device)
    )

    return lr_feat_input_img, upsampler_input_img


class CompleteUpsampler(nn.Module):
    def __init__(
        self,
        feature_type: FeatureType,
        chk_or_cfg: str | UpsamplerConfig,
        denoiser_chk: str | None = None,
        autoencoder_chk_or_cfg: str | AutoencoderConfig | None = None,
        device: str = "cpu",
        add_flash_attn: bool = True,
        to_eval: bool = False,
        to_half: bool = False,
    ) -> None:
        super().__init__()
        self.feature_type: FeatureType = feature_type
        # Load base DINOv2 model
        self.dv2_model = PretrainedViTWrapper(MODEL_MAP[feature_type], add_flash_attn=add_flash_attn, device=device)
        self.denoiser: Denoiser | None = None
        self.autoencoder: Autoencoder | None = None

        # Load or initialise optional models
        if denoiser_chk is None and feature_type != "FEATUP":
            self.denoiser = get_denoiser(None, device, to_eval, to_half)
        elif denoiser_chk is not None and feature_type != "FEATUP":
            self.denoiser = get_denoiser(denoiser_chk, device, to_eval, to_half)

        if isinstance(autoencoder_chk_or_cfg, str):
            self.autoencoder = get_autoencoder(autoencoder_chk_or_cfg, None, device, to_eval, to_half)
        elif isinstance(autoencoder_chk_or_cfg, AutoencoderConfig):
            self.autoencoder = get_autoencoder(None, autoencoder_chk_or_cfg, device, to_eval, to_half)

        # Load or initialise upsampler
        if isinstance(chk_or_cfg, str):
            self.upsampler = get_upsampler(chk_or_cfg, None, device, to_eval, to_half)
        else:
            self.upsampler = get_upsampler(None, chk_or_cfg, device, to_eval, to_half)

        self.device = device
        self.to_eval = to_eval
        if self.to_eval:
            self = self.eval()
        self.to_half = to_half

    def transform_image(self, image: np.ndarray | Image.Image | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return transform_image(image, self.device, self.to_half)

    def get_lr_feats(
        self,
        image: np.ndarray | Image.Image | torch.Tensor,
        output_norm: bool = True,
        transform_input: bool = True,
        existing_pca: PCAUnprojector | None = None,
    ) -> torch.Tensor:
        if transform_input:
            lr_feat_input_img, _ = self.transform_image(image)
        else:
            assert isinstance(image, torch.Tensor), "Image must be tensor if transform_input is false"
            lr_feat_input_img = image

        match self.feature_type:
            case "FEATUP":
                k = self.upsampler.n_ch_in
                lr_feats, _ = get_lr_featup_feats_and_pca(
                    self.dv2_model, [lr_feat_input_img], n_feats_in=k, existing_pca=existing_pca
                )
            case "LOFTUP_FULL":
                assert self.denoiser is not None
                dv2_feats = self.dv2_model.forward_features(lr_feat_input_img, make_2D=True)
                lr_feats = self.denoiser.forward_(dv2_feats)
            case "LOFTUP_COMPRESSED":
                assert self.denoiser is not None
                assert self.autoencoder is not None
                dv2_feats = self.dv2_model.forward_features(lr_feat_input_img, make_2D=True)
                denoised_feats = self.denoiser.forward_(dv2_feats)
                denoised_feats = F.normalize(denoised_feats, p=1, dim=1)
                lr_feats = self.autoencoder.encoder(denoised_feats)
            case _:
                raise Exception(f"Unsupported feature type {_}")

        if output_norm:
            lr_feats = F.normalize(lr_feats, p=1, dim=1)
        return lr_feats

    def get_hr_feats(
        self, upsampler_img: torch.Tensor, lr_feats: torch.Tensor, cast_to: torch.dtype = torch.float32
    ) -> torch.Tensor:
        with torch.autocast(self.device, cast_to):
            hr_feats = self.upsampler(upsampler_img, lr_feats)
        return hr_feats

    def forward(
        self, image: np.ndarray | Image.Image | torch.Tensor, existing_pca: PCAUnprojector | None = None
    ) -> torch.Tensor:
        lr_feat_input_img, upsampler_input_img = self.transform_image(image)
        lr_feats = self.get_lr_feats(lr_feat_input_img, True, False, existing_pca=existing_pca)
        dtype = torch.float16 if self.device else torch.float32
        return self.get_hr_feats(upsampler_input_img, lr_feats, dtype)


# OLD FUNCTIONS
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
        n_ch_deep=expr.n_ch_hidden,
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
    device: str = "cuda:0",
    fit_3d: bool = True,
    n_imgs_for_red: int = 50,
    n_ch_in: int = 64,
    n_batch_lr: int = 50,
    existing_pca: PCAUnprojector | None = None,
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
        existing_pca=existing_pca,
    )
    reduced_tensor = F.normalize(reduced_tensor, p=1, dim=1)

    hr_feats: torch.Tensor
    with torch.autocast("cuda", torch.float16):
        hr_feats = upsampler(inp_img, reduced_tensor)
    return hr_feats

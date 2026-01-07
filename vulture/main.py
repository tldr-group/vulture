from typing import Any, Mapping, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

import numpy as np

from vulture.models.external.alibi_vit_wrapper import AlibiVitWrapper
from vulture.models.external.vit_wrapper import add_flash_attention
from vulture.models import (
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
from vulture.feature_prep import (
    get_lr_feats,
    PCAUnprojector,
    get_lr_featup_feats_and_pca,
)
from vulture.utils import (
    expriment_from_json,
    closest_crop,
    convert_image,
    Experiment,
)

TransformFn = Callable[[np.ndarray | Image.Image | torch.Tensor, str, bool, int], tuple[torch.Tensor, torch.Tensor]]


def default_image_transform(
    image: np.ndarray | Image.Image | torch.Tensor, device: str, to_half: bool = False, patch_size: int = 14
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take input image and transform for both original feature model (closest crop to multiple of patch length, norm)
    and for upsampler image (norm).

    Args:
        image (np.ndarray | Image.Image | torch.Tensor): image whose features we wish to upsample
        device (str): CUDA device string
        to_half (bool, optional): put image in half precision. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: transformed image for feature model, transformed image for upsampler
            These are both (1,C,H,W).
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")

    if isinstance(image, torch.Tensor):
        _h, _w = image.shape[-2:]
        batch = len(image.shape) < 4
    else:
        _h, _w = image.height, image.width
        batch = True

    tr = closest_crop(_h, _w, patch_size=patch_size)
    lr_feat_input_img = convert_image(
        image,
        tr,
        batch=batch,
        to_gpu=device != "cpu",
        device_str=device,
        to_half=to_half,
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
        dino_chk: str | None = None,
        device: str = "cpu",
        add_flash_attn: bool = False,
        to_eval: bool = False,
        to_half: bool = False,
        transform_fn: TransformFn = default_image_transform,
    ) -> None:
        """Feature upsampler wrapper class.

        Args:
            feature_type (FeatureType): which featureset to use when upsampling. One of "FEATUP" (trained to match HR
                featup-implict ground truths [1]) or "LOFTUP"/"LOFTUP" [2] (compressed=compressed via supplied autoencoder)
            chk_or_cfg (str | UpsamplerConfig): path to .pt checkpoint (which should contain the upsampler config as a json
                alongside weights), or UpsamplerConfig for fresh model
            denoiser_chk (str | None, optional): path to .pt checkpoint of feature denoiser from [3]. Defaults to None.
            autoencoder_chk_or_cfg (str | AutoencoderConfig | None, optional): path to .pt checkpoint of autoencoder feature
                compressor (again must contain config alongside weights). The autoencoder needs to be trained for each low-res
                featureset you wish to uspample. Defaults to None.
            device (str, optional): CUDA device string to load model(s) onto. Defaults to "cpu".
            add_flash_attn (bool, optional): whether to add flash attention to the base feature model. Defaults to True.
            to_eval (bool, optional): put model(s) in eval mode. Defaults to False.
            to_half (bool, optional): put model(s) in half precision. Required for flash-attention. Defaults to False.

        - [1] S. Fu _et al._, "FeatUp: A Model-Agnostic Framework for Features at Any Resolution" (2024), ICLR, https://arxiv.org/abs/2403.10516
        - [2] H. Huang _et al._, "LoftUp: A Coordinate-Based Feature Upsampler for Vision Foundation Models", ICCV, https://arxiv.org/abs/2504.14032
        - [3] J. Yang _et al._, "Denoising Vision Transformers" (2024), ECCV, https://arxiv.org/abs/2401.02957
        """
        super().__init__()
        self.feature_type: FeatureType = feature_type
        # Load base DINOv2 model
        self.dv2_model: PretrainedViTWrapper
        if feature_type == "ALIBI_COMPRESSED":
            assert dino_chk is not None, "Must supply Alibi finetuned DINOv2 checkpoint"
            self.dv2_model = AlibiVitWrapper(MODEL_MAP[feature_type], add_flash_attn=add_flash_attn, device=device)
        else:
            self.dv2_model = PretrainedViTWrapper(MODEL_MAP[feature_type], add_flash_attn=add_flash_attn, device=device)
        # Apply weights
        if dino_chk is not None:
            weights = torch.load(dino_chk, map_location=device, weights_only=True)
            self.dv2_model.load_state_dict(weights)

        self.denoiser: Denoiser | None = None
        self.autoencoder: Autoencoder | None = None

        needs_denoiser = ("LOFTUP_FULL", "LOFTUP_COMPRESSED")
        # Load or initialise optional models
        if denoiser_chk is None and feature_type in needs_denoiser:
            self.denoiser = get_denoiser(None, device, to_eval, to_half)
        elif denoiser_chk is not None and feature_type in needs_denoiser:
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
        if self.to_half:
            self = self.half()

        if self.to_half is False and add_flash_attn is True:
            raise Exception("Flash attention requires half precision. Set to_half=True.")

        self.transform_fn = transform_fn

    def transform_image(
        self, image: np.ndarray | Image.Image | torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Take input image and transform for both original feature model (closest crop to multiple of patch length, norm)
        and for upsampler image (norm).

        Args:
            image (np.ndarray | Image.Image | torch.Tensor): image whose features we wish to upsample
            device (str): CUDA device string
            to_half (bool, optional): put image in half precision. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: transformed image for feature model, transformed image for upsampler.
                These are (1,C,H,W).
        """
        return self.transform_fn(image, self.device, self.to_half, self.dv2_model.patch_size, **kwargs)

    def get_lr_feats(
        self,
        image: np.ndarray | Image.Image | torch.Tensor,
        output_norm: bool = True,
        transform_input: bool = True,
        existing_pca: PCAUnprojector | None = None,
        n_imgs_pca: int = 50,
        n_batch: int = 50,
    ) -> torch.Tensor:
        """Get low-res features from base feature model for image. This does one of the following (based on $self.feature_type):

        FEATUP: compute shared PCA of features (from, say, DINOv2) of jittered (small transforms: pads, zooms, flips) of input
            image down to $k dimensions (usually 128, 64 or 32) -> $PCA. Compute features from base feature model -> $F then
            apply $PCA to $F to get $F'. Return $F', which is (1, $k, N_th, N_tw)
        LOFTUP: get features $F from base feature model, apply feature denoiser [3], return $F' which is (1, D, N_th, N_tw)
        LOFTUP_COMPRESSED: get features $F from base feature model, apply feature denoiser [3], compress with autoencoder to
            $k dims (usually 48). Return $F' which is (1, $k, N_th, N_tw)

        Terms:
            D = hidden dimension of base feature model (384 for ViT-S / DINOv2-ViT-S)
            N_th/w = number of patch tokens in the height/width dimension. These are height // patch_size and width // patch_size
                for the transform of the image (closest crop).

        Args:
            image (np.ndarray | Image.Image | torch.Tensor): input image. Can be already_transformed if $transform_input=False
            output_norm (bool, optional): whether to apply L1 norm to output. Defaults to True.
            transform_input (bool, optional): whether to transform the input (i.e closest crop). Defaults to True.
            existing_pca (PCAUnprojector | None, optional): existing $PCA that can be supplied instead of computing from scratch.
                Useful if applying over batch of images. Only needed for FEATUP. Defaults to None.
            n_imgs_pca (int, optional): number of transforms to use in FEATUP shared PCA. Defaults to 50.
            n_batch (int, optional): batch size of transformed feature computation in FEATUP shared PCA. Turn this down if running
                into memory issues for large images. Defaults to 50.

        Raises:
            Exception: if invalid feature type supplied

        Returns:
            torch.Tensor: low-res features for upsampling, shape (1, n, N_th, N_tw)
        """
        if transform_input:
            lr_feat_input_img, _ = self.transform_image(image)
        else:
            assert isinstance(image, torch.Tensor), "Image must be tensor if transform_input is false"
            lr_feat_input_img = image

        match self.feature_type:
            case "FEATUP":
                k = self.upsampler.n_ch_in
                lr_feats, _ = get_lr_featup_feats_and_pca(
                    self.dv2_model,
                    [lr_feat_input_img],
                    n_feats_in=k,
                    existing_pca=existing_pca,
                    n_imgs=n_imgs_pca,
                    n_batch=n_batch,
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
            case "ALIBI_COMPRESSED":
                assert self.autoencoder is not None
                dv2_feats = self.dv2_model.forward_features(lr_feat_input_img, make_2D=True)
                dv2_feats = F.normalize(dv2_feats, p=1, dim=1)
                lr_feats = self.autoencoder.encoder(dv2_feats)
            case _:
                raise Exception(f"Unsupported feature type {_}")

        if output_norm:
            lr_feats = F.normalize(lr_feats, p=1, dim=1)
        return lr_feats

    def get_hr_feats(
        self,
        upsampler_img: torch.Tensor,
        lr_feats: torch.Tensor,
        cast_to: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get upsampled features for input image $upsampler_img and low-res features $lr_feats.

        Args:
            upsampler_img (torch.Tensor): (1, C, H, W) image
            lr_feats (torch.Tensor): (1, n, N_th, N_tw) low-res feautres of image
            cast_to (torch.dtype, optional): output type of high-res-features. Defaults to torch.float32.

        Returns:
            torch.Tensor: (1, n, H, W) high-res features
        """
        with torch.autocast(self.device, cast_to):
            hr_feats = self.upsampler.forward(upsampler_img, lr_feats)
        return hr_feats

    def forward(
        self,
        image: np.ndarray | Image.Image | torch.Tensor,
        existing_pca: PCAUnprojector | None = None,
        n_imgs_pca: int = 50,
        n_batch: int = 50,
    ) -> torch.Tensor:
        """Get upsampled features for input image $upsampler_img, computing low-res features first.

        Args:
            image (np.ndarray | Image.Image | torch.Tensor): input image. Can be already_transformed if $transform_input=False
            existing_pca (PCAUnprojector | None, optional): existing $PCA that can be supplied instead of computing from scratch.
                Useful if applying over batch of images. Only needed for FEATUP. Defaults to None.
            n_imgs_pca (int, optional): number of transforms to use in FEATUP shared PCA. Defaults to 50.
            n_batch (int, optional): batch size of transformed feature computation in FEATUP shared PCA. Turn this down if running
                into memory issues for large images. Defaults to 50.

        Returns:
            torch.Tensor: (1, n, H, W) high-res features
        """
        lr_feat_input_img, upsampler_input_img = self.transform_image(image)
        lr_feats = self.get_lr_feats(
            lr_feat_input_img, True, False, existing_pca=existing_pca, n_imgs_pca=n_imgs_pca, n_batch=n_batch
        )
        dtype = torch.float16 if self.to_half else torch.float32
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
    cfg_path: str | None,
    device: str | torch.device = "cuda:0",
) -> tuple[FeatureUpsampler, Experiment | UpsamplerConfig]:
    obj: Mapping[str, Any] = torch.load(chk_path, weights_only=True, map_location=device)

    expr: Experiment | UpsamplerConfig
    if "config" in obj:
        expr = UpsamplerConfig(**obj["config"])
        state_dict = obj["weights"]
    else:
        assert cfg_path is not None
        expr = expriment_from_json(cfg_path)
        state_dict = obj

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
    upsampler.load_state_dict(state_dict)
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

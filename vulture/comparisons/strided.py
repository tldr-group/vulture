import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

from vulture.comparisons.vision_transformer import DinoVisionTransformer

from types import MethodType
from typing import Callable, Literal


class Patch:
    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: tuple[int, int]) -> Callable:
        """Creates a method for position encoding interpolation, used to overwrite
        the original method in the DINO/DINOv2 vision transformer.
        Taken from https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py,
        added some bits from the Dv2 code in.

        :param patch_size: patch size of the model.
        :type patch_size: int
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :type Tuple[int, int]
        :return: the interpolation method
        :rtype: Callable
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            previous_dtype = x.dtype
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            pos_embed = self.pos_embed.float()
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0: float = 1 + (w - patch_size) // stride_hw[1]
            h0: float = 1 + (h - patch_size) // stride_hw[0]
            assert w0 * h0 == npatch, f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
            #                               stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = F.interpolate(
                patch_pos_embed.reshape(1, int(sqrt(N)), int(sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / sqrt(N), h0 / sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

        return interpolate_pos_encoding


DinoVersions = Literal["dino", "dinov2"]
DinoTypes = Literal["vits14", "vitb14", "vits14_reg", "vitb14_reg"]


class StridedDv2(nn.Module):
    def __init__(
        self,
        dino_ver: DinoVersions,
        dino_type: DinoTypes,
        stride: int,
    ) -> None:
        super().__init__()

        self.model: DinoVisionTransformer = self.get_model(dino_ver, dino_type)

        self.original_patch_size = 14
        self.has_reg_tokens = "reg" in dino_type

        self.set_model_stride(self.model, stride)

    def get_model(self, version: DinoVersions, type_: DinoTypes) -> DinoVisionTransformer:
        hub_path = f"facebookresearch/{version}"
        if version == "dino":
            hub_path += ":main"
        return torch.hub.load(hub_path, f"{version}_{type_}")

    def set_model_stride(self, dino_model: nn.Module, stride_l: int, verbose: bool = False) -> None:
        """Create new positional encoding interpolation method for $dino_model with
        supplied $stride, and set the stride of the patch embedding projection conv2D
        to $stride.

        :param dino_model: Dv2 model
        :type dino_model: DinoVisionTransformer
        :param new_stride: desired stride, usually stride < original_stride for higher res
        :type new_stride: int
        :return: None
        :rtype: None
        """

        new_stride_pair = torch.nn.modules.utils._pair(stride_l)
        self.stride = new_stride_pair
        dino_model.patch_embed.proj.stride = new_stride_pair  # type: ignore
        if verbose:
            print(f"Setting stride to ({stride_l},{stride_l})")

        dino_model.interpolate_pos_encoding = MethodType(  # type: ignore
            Patch._fix_pos_enc(self.original_patch_size, new_stride_pair),
            dino_model,
        )  # typed ignored as they can't type check reassigned methods (generally is poor practice)

    def forward(self, x: torch.Tensor, make_spatial: bool = True) -> torch.Tensor:
        out = self.model.forward_features(x)
        feats = out["x_norm_patchtokens"]
        if make_spatial:
            stride_l = self.stride[0]
            _, t, c = feats.shape
            _, _, h, w = x.shape
            n_patch_w: int = 1 + (w - self.original_patch_size) // stride_l
            n_patch_h: int = 1 + (h - self.original_patch_size) // stride_l

            feats = feats[0].T.reshape((c, n_patch_h, n_patch_w)).unsqueeze(0)
        return feats

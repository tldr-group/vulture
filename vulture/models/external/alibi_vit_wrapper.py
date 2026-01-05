import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Block, Attention

from typing import Type, Literal, Optional

from vulture.models.external.vit_wrapper import PretrainedViTWrapper


def get_distance_matrix(
    n_tokens_h: int,
    n_tokens_w: int,
    n_reg_tokens: int = 4,
    metric: str = "euclidean",
    normalize: bool = True,
    wrap: bool = False,
    add_cls: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # TODO: is this (H,W) or (W,H) - and which is right?
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(n_tokens_h, device=device),
            torch.arange(n_tokens_w, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)  # (N, 2)
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    diff = diff.abs()

    if wrap:
        diff[..., 0] = torch.minimum(diff[..., 0], n_tokens_h - diff[..., 0])
        diff[..., 1] = torch.minimum(diff[..., 1], n_tokens_w - diff[..., 1])

    if metric == "euclidean":
        D = torch.sqrt((diff**2).sum(-1))
    elif metric == "manhattan":
        D = diff.sum(-1)
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")

    if normalize:
        D /= D.max()
    D *= -1

    n_extra_tokens = int(add_cls) + n_reg_tokens
    # timm prepends [CLS] + [REG]
    D = F.pad(D, (n_extra_tokens, 0, n_extra_tokens, 0), mode="constant")

    return D.to(device=device, dtype=dtype)


AlibiSlopeType = Literal["fixed", "learned", "constant"]


def get_alibi_slope(
    num_heads: int, slope_type: AlibiSlopeType = "constant", device: str = "cpu"
) -> torch.Tensor | nn.Parameter:
    m: torch.Tensor | nn.Parameter
    match slope_type:
        case "fixed":
            xs = (2**8) ** (1 / num_heads)
            m = torch.tensor([1 / xs ** (i + 1) for i in range(num_heads)], device=device)
        case "learned":
            m = torch.rand(num_heads, device=device)
        case "constant":
            m = torch.ones(num_heads, device=device)
        case _:
            raise ValueError(f"Unknown slope type {type}")

    m = m.unsqueeze(-1).unsqueeze(-1)
    if slope_type == "learned":
        m = nn.Parameter(m)
    else:
        m.requires_grad = False
    return m


class DistanceMatrixWrapper(nn.Module):
    def __init__(
        self,
        n_tokens_h: int,
        n_tokens_w: int,
        n_reg_tokens: int = 4,
        metric: str = "euclidean",
        normalize: bool = True,
        wrap: bool = True,
        add_cls: bool = True,
    ) -> None:
        super().__init__()

        self.n_tokens_h = n_tokens_h
        self.n_tokens_w = n_tokens_w
        self.n_reg_tokens = n_reg_tokens
        self.metric = metric
        self.normalize = normalize
        self.wrap = wrap
        self.add_cls = add_cls

        # self.matrix: torch.Tensor | None = None

        self.update(
            n_tokens_h,
            n_tokens_w,
            n_reg_tokens=n_reg_tokens,
            metric=metric,
            normalize=normalize,
            wrap=wrap,
            add_cls=add_cls,
            force_update=True,
        )

    def update(
        self,
        n_tokens_h: int,
        n_tokens_w: int,
        n_reg_tokens: int = 4,
        metric: str = "euclidean",
        normalize: bool = True,
        wrap: bool = True,
        add_cls: bool = True,
        force_update: bool = False,
    ) -> None:
        is_stale = False
        for attr, val in (
            ("n_tokens_h", n_tokens_h),
            ("n_tokens_w", n_tokens_w),
            ("n_reg_tokens", n_reg_tokens),
            ("metric", metric),
            ("normalize", normalize),
            ("wrap", wrap),
            ("add_cls", add_cls),
        ):
            if getattr(self, attr) != val:
                is_stale = True

        if not is_stale and not force_update:
            # nop if nothing has changed
            return

        try:
            device = self.matrix.device
            dtype = self.matrix.dtype
        except AttributeError:
            device = "cpu"
            dtype = torch.float32

        distance_matrix = get_distance_matrix(
            n_tokens_h,
            n_tokens_w,
            n_reg_tokens,
            wrap=wrap,
            metric=metric,
            normalize=normalize,
            add_cls=add_cls,
            device=device,
            dtype=dtype,
        )

        self.n_tokens_h = n_tokens_h
        self.n_tokens_w = n_tokens_w
        self.n_reg_tokens = n_reg_tokens
        self.metric = metric
        self.normalize = normalize
        self.wrap = wrap
        self.add_cls = add_cls
        self.register_buffer("matrix", distance_matrix, persistent=False)


class AlibiAttention(Attention):
    def __init__(
        self,
        distance_matrix: DistanceMatrixWrapper,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        slope_type: AlibiSlopeType = "constant",
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.distance_matrix = distance_matrix
        self.fused_attn = False
        self.set_alibi_slope(slope_type=slope_type)

        self.n_tokens_h = 16
        self.n_tokens_w = 16

        # self.is_enabled = True

    def set_alibi_slope(self, slope_type: AlibiSlopeType):
        m = get_alibi_slope(self.num_heads, slope_type=slope_type, device=self.qkv.weight.device)
        if isinstance(m, torch.Tensor):
            self.register_buffer("m", m, persistent=False)
        else:
            self.register_parameter("m", m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        bias = self.m * self.distance_matrix.matrix
        bias = bias.unsqueeze(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=bias
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn = attn + bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AlibiBlock(Block):
    def __init__(
        self,
        distance_matrix: DistanceMatrixWrapper,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.attn = AlibiAttention(
            distance_matrix,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )


class AlibiVitWrapper(PretrainedViTWrapper):
    def __init__(
        self,
        model_identifier: str = "vit_small_patch14_dinov2.lvd142m",
        stride: int = 14,
        add_flash_attn: bool = False,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        device: str = "cpu",
        slope_type: AlibiSlopeType = "constant",
        n_reg_tokens: int = 4,
        metric: str = "euclidean",
        normalize: bool = True,
        wrap: bool = True,
        add_cls: bool = True,
        **kwargs,
    ):
        distance_matrix = DistanceMatrixWrapper(
            n_tokens_h=16,
            n_tokens_w=16,
            n_reg_tokens=n_reg_tokens,
            metric=metric,
            normalize=normalize,
            wrap=wrap,
            add_cls=add_cls,
        )

        def block_fn_wrapper(dim: int, num_heads: int, **block_kwargs: dict) -> AlibiBlock:
            return AlibiBlock(distance_matrix=distance_matrix, dim=dim, num_heads=num_heads, **block_kwargs)

        super().__init__(
            model_identifier=model_identifier,
            stride=stride,
            add_flash_attn=add_flash_attn,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            device=device,
            block_fn=block_fn_wrapper,
            **kwargs,
        )
        # self.model.pos_embed = None
        self.distance_matrix = distance_matrix
        # self.model.pos_embed.requires_grad = False  # freeze pos embedding
        self.slope_type = slope_type

        # self.embed_dim = 24

        for blk in self.model.blocks:
            blk: AlibiBlock
            blk.attn.set_alibi_slope(slope_type=self.slope_type)

    def forward(self, x: torch.Tensor):
        # TODO: set alibi distance matrix size
        return self.forward_features(x, make_2D=True), None

    def forward_features(self, x: torch.Tensor, make_2D: bool = False, add_reg: bool = False, **kwargs) -> torch.Tensor:
        assert self.model.pos_embed is not None
        b, _, h, w = x.shape
        p = self.patch_size
        s = self.stride
        n_patch_h, n_patch_w = (h - p) // s + 1, (w - p) // s + 1

        self.distance_matrix.update(n_patch_h, n_patch_w)

        feats = super().forward_features(x, make_2D, add_reg)
        return feats

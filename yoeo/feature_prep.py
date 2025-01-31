import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import random
from sklearn.decomposition import PCA

"""
This code is loosely adapted/directly taken from FeatUp [1, 2], whose high-res implict features I use as a training target
for my upsampler. When training their implict upsampler per-image they prepare a dataset of features (from, say, DINO) of
'jitters'=transforms of the original input image, then compute and apply a shared PCA over these jittered features. This is part
of their NeRF analogy - "that multiview consistency of low-resolution signals can supervise the construction of 
high-resolution signals" [2]. Crucially, this shared PCA changes both the number of channels and the distribution of 
features when compared to the features produced by DINO for the untransformed image in both low and high-res.

That these features are subtly different makes the simple, image-guided feature upsampling task I want my (lightweight) CNN to 
achieve much more difficult, in that it now has to do two things - the first being to upsample based on image features and
the second being to convert between these two spaces. This means a contractive approach, which goes from the original
ViT-style features (B, N_patch_H, N_patch_W, 384) to the high-res FeatUp features (B, H, W, 128) is difficult to learn.

Converting from the LR-DINO features -> LR-FeatUp features (i.e having the shared PCA applied) with a separate preparatory network 
also seemed to be difficult. I can only assume that the FeatUp PCA'd LR features have information from multiple queries to our
DINO(v2) 'oracle', and that the affect of those transformations on the shared PCA (and therefore on the LR features) is difficult
to predict without learning to some degree how DINO produces is features, which is a) difficult for a small CNN and b) what I
wanted to avoid with the upsampler to begin with.

The upshot of all this is that the best way to upsample the features is to convert them exactly the same way FeatUp does to 
produce its training LR-features and, specifically, its LR preview features. This means computing the DINO(v2) features for
N transforms of the original image, computing a shared PCA and applying that PCA to the features of the original, untransformed
image. Empirically these share a similar enough distribution (observed in PCAs of training data) that the upsampler can be 
trained. To generate the training set, I simply take the preview features from each implict FeatUp run and store it with
the high resolution features. When I need to apply the upsampler unseen, I need to reproduce the jitter-dataset-PCA process
before I can upsample - this is now the slowest part of the process. I can speed it up by adding 'flash-attn'[3] to DINO(v2)
s.t for large images (>1000x1000) I am bottlenecked by the cost of computing the augmentations of the image. This then motivates
the propagator to quickly extract the LR-features so they can be input into the upsampler.


[1] https://github.com/mhamilton723/FeatUp
[2] Fu, S. et al, 'FeatUp: A Model-Agnostic Framework for Features at Any Resolution', https://arxiv.org/abs/2403.10516
[3] https://github.com/Dao-AILab/flash-attention
[4] Dao, T., 'FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning', https://tridao.me/publications/flash2/flash2.pdf
"""


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

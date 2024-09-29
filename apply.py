import torch
from PIL import Image

from utils import to_numpy, do_2D_pca, add_flash_attention, load_image, closest_crop
from model import Combined


DEVICE = "cuda:0"
dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dv2 = add_flash_attention(dv2)
dv2 = dv2.eval().half().to(DEVICE)


upsampler = torch.load("upsample.pth")
"""
Combined(
    14, n_ch_img=3, n_ch_in=128, n_ch_downsample=64, k_up=3, feat_weight=0.25
)
"""
upsampler = upsampler.eval().half().to(DEVICE)

path = "data/compare/3.jpg"
_img = Image.open(path)
_h, _w = _img.height, _img.width
tr = closest_crop(_h, _w)
img, original = load_image(path, tr)

_, _, h, w = img.shape


n_patch_w: int = 1 + (w - 14) // 14
n_patch_h: int = 1 + (h - 14) // 14
feats_dict: dict = dv2.forward_features(img)  # type: ignore
feats = feats_dict["x_norm_patchtokens"]
b, _, c = feats.shape
feats = feats.reshape(b, c, n_patch_h, n_patch_w)
print(feats.shape)
reduced = do_2D_pca()
print(reduced.shape)

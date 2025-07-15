import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from random import seed

import numpy as np
from time import time_ns

from yoeo.utils import to_numpy, do_2D_pca, closest_crop, convert_image
from yoeo.main import get_dv2_model, get_upsampler_and_expr, get_hr_feats
from yoeo.feature_prep import get_lr_feats, project
from yoeo.comparisons.online_denoiser import Denoiser
from yoeo.comparisons.autoencoder import get_autoencoder
from yoeo.datasets.learn_remap_LU_feats import vis

torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()


SEED = 10672
np.random.seed(SEED)
torch.manual_seed(SEED)
seed(SEED)

DEVICE = "cuda:0"

dv2 = get_dv2_model(False, to_half=False, add_flash=False, device=DEVICE)
# dv2 = get_dv2_model(True, to_half=False, add_flash=False, device=DEVICE)
dv2 = dv2.to(DEVICE)

denoiser = Denoiser(feat_dim=384).to(DEVICE)
denoiser_weights = torch.load("yoeo/comparisons/vit_small_patch14_reg4_dinov2.lvd142m.pth")
denoiser.load_state_dict(denoiser_weights["denoiser"])

autoencoder = get_autoencoder("trained_models/dac_dv2_denoised_e500.pth", DEVICE)
autoencoder = autoencoder

# model_path = "experiments/old/280525_up_lu_fixed_feats_half/best.pth"
# model_path = "trained_models/e180_full_dv2.pth"
# cfg_path = "yoeo/models/configs/upsampler_full_dv2.json"
# model_path = "trained_models/e5000_full_fit_reg.pth"
# cfg_path = "yoeo/models/configs/combined_no_shift.json"

# model_path = "trained_models/e5000_fit_reg_f32.pth"
# model_path = "experiments/current/best.pth"
# cfg_path = "yoeo/models/configs/upsampler_FU_ch32.json"

# model_path = "trained_models/e5000_fit_reg_f64.pth"
# cfg_path = "yoeo/models/configs/upsampler_fewer_features.json"

# model_path = "experiments/current/best.pth"
# # model_path = "trained_models/e256_full_dv2.pth"
# cfg_path = "yoeo/models/configs/upsampler_LU_narrow.json"

model_path = "experiments/current/best.pth"
cfg_path = "yoeo/models/configs/upsampler_LU_compressed.json"

upsampler, expr = get_upsampler_and_expr(model_path, cfg_path, device=DEVICE)
upsampler = upsampler.eval()

path = "data/compare/accom.png"
img = Image.open(path).convert("RGB")
# img = img.resize((img.width // 2, img.height // 2))

tr = closest_crop(img.height, img.width, 14, True)


inp_img = (
    TF.normalize(
        TF.pil_to_tensor(img).to(torch.float32),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    .unsqueeze(0)
    .to(DEVICE)
)
inp_img_dino = convert_image(img, tr, to_half=False, device_str=DEVICE)
# print(inp_img_dino.shape)

torch.cuda.reset_peak_memory_stats(DEVICE)  # s.t memory is accurate
torch.cuda.synchronize(DEVICE)  # s.t time is accurate


def _to_MB(x: int) -> float:
    return x / (1024**2)


def _to_s(t: int) -> float:
    return t / 1e9


torch.cuda.reset_peak_memory_stats(DEVICE)  # s.t memory is accurate
torch.cuda.synchronize(DEVICE)  # s.t time is accurate

m0 = torch.cuda.max_memory_allocated(DEVICE)
t0 = time_ns()

with torch.no_grad():
    lr_feats = project(inp_img_dino, dv2, fit3d=False)
    # lr_feats, _ = get_lr_feats(dv2, [inp_img_dino], 50, fit3d=True, n_feats_in=expr.n_ch_in)
    # lr_feats = lr_feats.to(DEVICE)
    # lr_feats = F.normalize(lr_feats, p=1, dim=1)

    lr_feats = lr_feats.permute((0, 2, 3, 1))
    lr_feats = denoiser.forward(lr_feats, return_channel_first=True)

    lr_feats = F.normalize(lr_feats, p=1, dim=1)
    lr_feats = autoencoder.encoder(lr_feats)
    # lr_feats = lr_feats.to(torch.float16)
    # inp_img = inp_img.to(torch.float16)
    lr_feats = F.normalize(lr_feats, p=1, dim=1)
    with torch.autocast(DEVICE, torch.float32):
        hr_feats = upsampler(inp_img, lr_feats)

torch.cuda.synchronize(DEVICE)  # s.t time is accurate
t1 = time_ns()
m1 = torch.cuda.max_memory_allocated(DEVICE)
print(f"{hr_feats.shape} in {_to_s(t1 - t0):.3f}s and {_to_MB(m1 - m0):.0f}MB {hr_feats.dtype}")


vis("tmp/test.png", img, lr_feats, hr_feats, None, True)


# W = 1500
# _w, _h = img.size
# sf = _w / 448
# new_h = int(_h / sf)
# print(new_h)

# img = img.resize((448, new_h))
# oy = (new_h - 448) / 2
# img = img.crop((0, oy, 448, oy + 448))

# hr_feats = get_hr_feats(img, dv2, upsampler, DEVICE, fit_3d=False, n_ch_in=expr.n_ch_in)


# reduced_tensor, _ = get_lr_feats(dv2, [inp_img_dino], 50, fit3d=True, n_feats_in=64)
# reduced_tensor = reduced_tensor.to(DEVICE)
# reduced_tensor = F.normalize(reduced_tensor, p=1, dim=1)

# with torch.autocast("cuda", torch.float16):
#     hr_feats = upsampler(inp_img, reduced_tensor)

# hr_feats_np = to_numpy(hr_feats)
# reduced_hr = do_2D_pca(hr_feats_np, 3).transpose((2, 0, 1))
# # reduced_hr = hr_feats_np[:3]
# c, h, w = reduced_hr.shape
# print(reduced_hr.shape)
# reduced_hr_flat = reduced_hr.reshape((c, h * w)).T
# reduced_hr_rescaled = MinMaxScaler(clip=True).fit_transform(reduced_hr_flat)
# reduced_hr = reduced_hr_rescaled.reshape((h, w, c))
# plt.imsave(
#     "tmp/test.png",
#     reduced_hr,
# )


# # dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
# dv2 = torch.hub.load("ywyue/FiT3D", "dinov2_reg_small_fine")
# dv2 = add_flash_attention(dv2)
# dv2 = dv2.eval().to(DEVICE).half()

# model_path = "trained_models/e1000_reg.pth"

# upsampler_weights = torch.load(model_path, weights_only=True, map_location="cuda:0")

# """
# Combined(
#     14, n_ch_img=3, n_ch_in=128, n_ch_downsample=64, k_up=3, feat_weight=0.25
# )
# """
# # upsampler: Combined = upsampler.eval().to(DEVICE)
# upsampler = FeatureUpsampler(
#     14,
#     n_ch_img=3,
#     n_ch_in=64,
#     n_ch_out=16,
#     n_ch_downsample=64,
#     k_up=3,
#     feat_weight=-1,
#     padding_mode="replicate",
# )
# upsampler.load_state_dict(upsampler_weights)
# upsampler = upsampler.eval().to(DEVICE)


# L = 322 * 2  # 2 * 224
# _img = Image.open(path).convert("RGB")  # .resize((L, L))
# _h, _w = _img.height, _img.width
# tr = closest_crop(_h, _w)


# img, original = load_image(path, tr, to_half=True)

# inp_img = (
#     TF.normalize(
#         TF.pil_to_tensor(_img).to(torch.float32),
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225],
#     )
#     .unsqueeze(0)
#     .to(DEVICE)
# )

# _, _, h, w = img.shape

# HALF = True
# if HALF:
#     img = img.half()

# original = original.convert("RGB")


# torch.cuda.reset_peak_memory_stats(img.device)  # s.t memory is accurate
# torch.cuda.synchronize(img.device)  # s.t time is accurate


# def _to_MB(x: int) -> float:
#     return x / (1024**2)


# def _to_s(t: int) -> float:
#     return t / 1e9


# m0 = torch.cuda.max_memory_allocated(img.device)
# t0 = time_ns()

# reduced_tensor, _ = get_lr_feats(dv2, [img], 50, fit3d=True, n_feats_in=64)
# torch.cuda.synchronize(img.device)
# t1 = time_ns()
# m1 = torch.cuda.max_memory_allocated(img.device)

# torch.cuda.reset_peak_memory_stats(img.device)  # s.t memory is accurate

# m2 = torch.cuda.max_memory_allocated(img.device)
# reduced_tensor = reduced_tensor.to(DEVICE)
# reduced_tensor = F.normalize(reduced_tensor, p=1, dim=1)

# with torch.autocast("cuda", torch.float16):
#     hr_feats = upsampler(inp_img, reduced_tensor)

# m3 = torch.cuda.max_memory_allocated(img.device)
# torch.cuda.synchronize(img.device)
# t2 = time_ns()


# featup_jbu = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=True).to(DEVICE)
# torch.cuda.reset_peak_memory_stats(img.device)  # s.t memory is accurate
# m4 = torch.cuda.max_memory_allocated(img.device)
# with torch.autocast("cuda", torch.float16):
#     jbu_feats = featup_jbu(img.to(torch.float32))
# m5 = torch.cuda.max_memory_allocated(img.device)
# jbu_feats = F.interpolate(jbu_feats, (h, w))
# jbu_feats_np = to_numpy(jbu_feats)
# print(jbu_feats.shape)

# reduced_jbu = do_2D_pca(jbu_feats_np, 3, post_norm="minmax")
# plt.imsave(
#     "test_jbu.png",
#     reduced_jbu,
# )

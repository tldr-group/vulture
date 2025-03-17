import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from yoeo.utils import to_numpy
from yoeo.main import get_dv2_model, get_upsampler_and_expr, get_hr_feats

torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()

DEVICE = "cuda:0"

dv2 = get_dv2_model(True, device=DEVICE)

model_path = "trained_models/e5000_fit_reg_f64.pth"
cfg_path = "yoeo/models/configs/upsampler_fewer_features.json"
upsampler, expr = get_upsampler_and_expr(model_path, cfg_path, device=DEVICE)

path = "data/apply/nmc.png"
img = Image.open(path).convert("RGB")

hr_feats = get_hr_feats(img, dv2, upsampler, DEVICE, n_ch_in=expr.n_ch_in)

hr_feats_np = to_numpy(hr_feats)
reduced_hr = hr_feats_np[:3]
c, h, w = reduced_hr.shape
reduced_hr_flat = reduced_hr.reshape((c, h * w)).T
reduced_hr_rescaled = MinMaxScaler(clip=True).fit_transform(reduced_hr_flat)
reduced_hr = reduced_hr_rescaled.reshape((h, w, c))
plt.imsave(
    "test.png",
    reduced_hr,
)


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

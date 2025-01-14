import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


from yoeo.utils import (
    to_numpy,
    do_2D_pca,
    add_flash_attention,
    load_image,
    closest_crop,
    resize_crop,
    get_lr_feats,
    Experiment,
)
from yoeo.models.model import Combined

torch.backends.cudnn.enabled = True


DEVICE = "cuda:0"
# dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dv2 = torch.hub.load("ywyue/FiT3D", "dinov2_reg_small_fine")
dv2 = add_flash_attention(dv2)
dv2 = dv2.eval().to(DEVICE).half()


# upsampler_weights = torch.load(
#     "apply_models/e4290_no_shift_no_feat_slow.pth", weights_only=True
# )

upsampler_weights = torch.load("apply_models/e5000_full_fit_reg.pth", weights_only=True)

featup_jbu = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=True).to(DEVICE)
"""
Combined(
    14, n_ch_img=3, n_ch_in=128, n_ch_downsample=64, k_up=3, feat_weight=0.25
)
"""
# upsampler: Combined = upsampler.eval().to(DEVICE)
upsampler = Combined(
    14,
    n_ch_img=3,
    n_ch_in=128,
    n_ch_out=128,
    n_ch_downsample=64,
    k_up=3,
    feat_weight=-1,
    padding_mode="replicate",
)
upsampler.load_state_dict(upsampler_weights)
upsampler = upsampler.eval().to(DEVICE)

# path = "data/compare/bar.JPEG"
path = "data/nz/accom.png"

# 500 ,375
L = 322 * 2  # 2 * 224
_img = Image.open(path).convert("RGB")  # .resize((L, L))
_h, _w = _img.height, _img.width
tr = closest_crop(_h, _w)
# norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# tr = T.Compose(
#     [
#         T.Resize((L, L)),
#         T.CenterCrop((L, L)),
#         T.ToTensor(),
#         norm,
#     ]
# )


img, original = load_image(path, tr, to_half=True)

inp_img = (
    TF.normalize(
        TF.pil_to_tensor(_img).to(torch.float32),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    .unsqueeze(0)
    .to(DEVICE)
)

_, _, h, w = img.shape

HALF = True
if HALF:
    img = img.half()

print(img.shape)
original = original.convert("RGB")

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


from time import time_ns

torch.cuda.reset_peak_memory_stats(img.device)  # s.t memory is accurate
torch.cuda.synchronize(img.device)  # s.t time is accurate


def _to_MB(x: int) -> float:
    return x / (1024**2)


def _to_s(t: int) -> float:
    return t / 1e9


m0 = torch.cuda.max_memory_allocated(img.device)
t0 = time_ns()

reduced_tensor = get_lr_feats(dv2, img, 50, fit3d=True)
torch.cuda.synchronize(img.device)
t1 = time_ns()
m1 = torch.cuda.max_memory_allocated(img.device)

torch.cuda.reset_peak_memory_stats(img.device)  # s.t memory is accurate

m2 = torch.cuda.max_memory_allocated(img.device)
reduced_tensor = reduced_tensor.to(DEVICE)
reduced_tensor = F.normalize(reduced_tensor, p=1, dim=1)

with torch.autocast("cuda", torch.float16):
    hr_feats = upsampler(inp_img, reduced_tensor)
#     flip_img = torch.flip(inp_img, (-1,))
#     flip_feats = torch.flip(reduced_tensor, (-1,))
#     hr_feats_flip_lr = upsampler(flip_img, flip_feats)
#     hr_feats_flip_ud = upsampler(
#         torch.flip(inp_img, (-2,)), torch.flip(reduced_tensor, (-2,))
#     )
# hr_stack = torch.cat(
#     (
#         hr_feats,
#         torch.flip(hr_feats_flip_lr, (-1,)),
#         torch.flip(hr_feats_flip_ud, (-2,)),
#     ),
#     dim=0,
# )
# print(hr_stack.shape)
# hr_feats = torch.mean(hr_stack, dim=0).unsqueeze(0)
# print(hr_feats.shape)

m3 = torch.cuda.max_memory_allocated(img.device)
torch.cuda.synchronize(img.device)
t2 = time_ns()

# print(
#     f"t_lr: {_to_s(t1 -t0)}s, t_up: {_to_s(t2-t1)}s, mem_lr: {_to_MB(m1 - m0):.3f}MB , mem_up: {_to_MB(m3 - m1):.3f}MB, mem_jbu: {_to_MB(m5 - m4):.3f}MB"
# )

# torch.tensor(reduced_np).permute((-1, 0, 1)).unsqueeze(0)


hr_feats_np = to_numpy(hr_feats)
# hr_feats_np = np.nan_to_num(hr_feats_np, nan=0)
reduced_hr = do_2D_pca(hr_feats_np, 3, post_norm="minmax")
# reduced_hr = hr_feats_np[:3, :, :].transpose((1, 2, 0))
print(hr_feats_np.shape)
plt.imsave(
    "test.png",
    reduced_hr,
)
# print([d.shape for d in downsamples])

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


from utils import (
    to_numpy,
    do_2D_pca,
    add_flash_attention,
    load_image,
    closest_crop,
    resize_crop,
    get_lr_feats,
    Experiment,
)
from model import Combined


DEVICE = "cuda:0"
dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dv2 = add_flash_attention(dv2)
dv2 = dv2.eval().to(DEVICE).half()


upsampler_weights = torch.load("apply_models/e490_no_shift.pth", weights_only=True)
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
    feat_weight=0.25,
    padding_mode="replicate",
)
upsampler.load_state_dict(upsampler_weights)
upsampler = upsampler.eval().to(DEVICE)

path = "data/compare/joined_image_crop.png"

# 500 ,375
L = 224
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


from time import time_ns

torch.cuda.reset_peak_memory_stats(img.device)  # s.t memory is accurate
torch.cuda.synchronize(img.device)  # s.t time is accurate


def _to_MB(x: int) -> float:
    return x / (1024**2)


def _to_s(t: int) -> float:
    return t / 1e9


start_m = torch.cuda.max_memory_allocated(img.device)
t0 = time_ns()

reduced_tensor = get_lr_feats(dv2, img, 25)
torch.cuda.synchronize(img.device)
t1 = time_ns()

reduced_tensor = reduced_tensor.to(DEVICE).to(torch.float32)
reduced_tensor = F.normalize(reduced_tensor, p=1, dim=1)
hr_feats = upsampler(inp_img, reduced_tensor)


end_m = torch.cuda.max_memory_allocated(img.device)
torch.cuda.synchronize(img.device)
t2 = time_ns()
print(
    f"t_lr: {_to_s(t1 -t0)}s, t_up: {_to_s(t2-t1)}s, mem: {_to_MB(end_m - start_m)}MB"
)

# torch.tensor(reduced_np).permute((-1, 0, 1)).unsqueeze(0)


hr_feats_np = to_numpy(hr_feats)
# hr_feats_np = np.nan_to_num(hr_feats_np, nan=0)
reduced_hr = do_2D_pca(hr_feats_np, 3, post_norm="minmax")
plt.imsave(
    "test.png",
    reduced_hr,
)
# print([d.shape for d in downsamples])

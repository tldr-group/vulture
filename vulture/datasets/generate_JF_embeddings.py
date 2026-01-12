import numpy as np
from vulture.models.external.autoencoder import get_autoencoder
from vulture.models.external.alibi_vit_wrapper import AlibiVitWrapper
from vulture.comparisons.jafar.jafar import JAFAR
from vulture.utils import resize_crop
from vulture.datasets.learn_remap_LU_feats import vis

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from datasets import load_dataset

from time import time

REMAP = True
DENOISE = True
DEVICE = "cuda:0"

np.random.seed(10001)
torch.random.manual_seed(10001)

ds = load_dataset("richwardle/reduced-imagenet", split="train")
ds = ds.shuffle()

torch.cuda.empty_cache()


model = AlibiVitWrapper(
    "vit_small_patch14_reg4_dinov2.lvd142m",
)
model = model.to(DEVICE)
weights = torch.load("trained_models/alibi_dv2_vits14_reg.pth", weights_only=True)
model.load_state_dict(weights)

autoencoder = get_autoencoder("trained_models/dac_alibi_dv2_e500_c24.pth", None, DEVICE)

patch_size = 14
dim = 24


kernel_size = patch_size
lr_size = 252 // patch_size  # 2 * 224 // patch_size
load_size = 252


upsampler = JAFAR(v_dim=dim).to(DEVICE).eval()
up_weights = torch.load("trained_models/jafar/alibi_c24.pth")["jafar"]
upsampler.load_state_dict(up_weights)

transform = resize_crop((load_size, load_size), (load_size, load_size))

N_CUTOFF = 5000
if __name__ == "__main__":
    N = len(ds)
    data_path = "data/imagenet_reduced"
    for i, dct in enumerate(ds):
        pil_img = dct["image"]
        start_t = time()
        normalized_img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lr_feats = model.forward_features(normalized_img_tensor, make_2D=True)
            compressed_lr = autoencoder.encoder(F.normalize(lr_feats, p=1, dim=1))

            hr_feats = upsampler(normalized_img_tensor, compressed_lr, (load_size, load_size))  # 1, dim, 224, 224

        data = {
            "lr_feats": compressed_lr,
            "dv2_lr_feats": lr_feats,
            "hr_feats": hr_feats,
        }
        pil_img.save(f"{data_path}/imgs/{i:05d}.png")
        torch.save(data, f"{data_path}/data_jf_reg/{i:05d}.pt")
        end_t = time()

        if i % 50 == 0:
            print(f"[{i:05d}/{N_CUTOFF}] in {end_t - start_t:03f}s")
            print(compressed_lr.shape, hr_feats.shape)
            vis(f"tmp/remap/remap_{i}.png", pil_img, compressed_lr, hr_feats, hr_feats)
        if i == N_CUTOFF:
            break

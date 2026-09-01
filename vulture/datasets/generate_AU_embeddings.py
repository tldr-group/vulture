from turtle import up
import numpy as np
from vulture.models.external.autoencoder import get_autoencoder
from vulture.models.external.vit_wrapper import PretrainedViTWrapper, MODEL_LIST
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

featurizer_class = "dinov2s_reg"
torch_hub_name = "loftup_dinov2s_reg"

model = PretrainedViTWrapper(MODEL_LIST[1], device=DEVICE, add_flash_attn=False)
model = model.to(DEVICE)

autoencoder = get_autoencoder("trained_models/dv2_c24.pth", None, DEVICE)


patch_size = 14
dim = 24


kernel_size = patch_size
lr_size = 252 // patch_size  # 2 * 224 // patch_size
load_size = 252


upsampler = torch.hub.load("wimmerth/anyup", "anyup_multi_backbone", use_natten=False)
upsampler = upsampler.to(DEVICE).eval()

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

            hr_feats = upsampler(normalized_img_tensor, compressed_lr)

        data = {
            "lr_feats": compressed_lr,
            "dv2_lr_feats": lr_feats,
            "hr_feats": hr_feats,
        }
        pil_img.save(f"{data_path}/imgs/{i:05d}.png")
        torch.save(data, f"{data_path}/data_au/{i:05d}.pt")
        end_t = time()

        if i % 50 == 0:
            print(f"[{i:05d}/{N_CUTOFF}] in {end_t - start_t:03f}s")
            print(compressed_lr.shape, hr_feats.shape)
            vis(f"tmp/remap/remap_{i}.png", pil_img, compressed_lr, hr_feats, hr_feats)
        if i == N_CUTOFF:
            break

import numpy as np
from yoeo.comparisons.loftup.upsamplers import norm, unnorm
from yoeo.comparisons.loftup.featurizers import get_featurizer
from yoeo.comparisons.loftup.utils import plot_feats

from yoeo.comparisons.online_denoiser import Denoiser
from yoeo.datasets.learn_remap_LU_feats import get_train_data, train_mlp, apply, vis

import torch
from PIL import Image
import torchvision.transforms as T

from torch.utils.data import DataLoader
from datasets import load_dataset

from time import time

REMAP = True
DENOISE = True
DEVICE = "cuda:1"

np.random.seed(10001)
torch.random.manual_seed(10001)

ds = load_dataset("richwardle/reduced-imagenet", split="train")
ds = ds.shuffle()

torch.cuda.empty_cache()

featurizer_class = "dinov2s_reg"
torch_hub_name = "loftup_dinov2s_reg"

model, patch_size, dim = get_featurizer(featurizer_class)
model = model.to(DEVICE)

denoiser = Denoiser(feat_dim=384).to(DEVICE)
denoiser_weights = torch.load("yoeo/comparisons/vit_small_patch14_reg4_dinov2.lvd142m.pth")
denoiser.load_state_dict(denoiser_weights["denoiser"])

kernel_size = patch_size
lr_size = 224 // patch_size  # 2 * 224 // patch_size
load_size = 224

upsampler = torch.hub.load("andrehuang/loftup", torch_hub_name, pretrained=True)
upsampler = upsampler.to(DEVICE).eval()


transform = T.Compose(
    [
        T.Resize(load_size, T.InterpolationMode.BILINEAR),
        T.CenterCrop(load_size),  # Depending on whether you want a center crop
        T.ToTensor(),
        norm,
    ]
)

N_CUTOFF = 4568
if __name__ == "__main__":
    N = len(ds)
    data_path = "data/imagenet_reduced"
    for i, dct in enumerate(ds):
        pil_img = dct["image"]
        start_t = time()
        normalized_img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            lr_feats = model(normalized_img_tensor)
            if DENOISE:
                lr_feats = lr_feats.permute((0, 2, 3, 1))
                lr_feats = denoiser.forward(lr_feats, return_channel_first=True)
            hr_feats = upsampler(lr_feats, normalized_img_tensor)  # 1, dim, 224, 224

        if REMAP:
            train, targ = get_train_data(lr_feats, hr_feats)
            mlp = train_mlp(train, targ, 256, 2000, lr=2e-3, verbose=False, device=DEVICE)
        with torch.no_grad():
            hr_feats = apply(mlp, hr_feats)

        data = {
            "lr_feats": lr_feats,
            "dv2_lr_feats": lr_feats,
            "hr_feats": hr_feats,
        }
        pil_img.save(f"{data_path}/imgs/{i:05d}.png")
        torch.save(data, f"{data_path}/data_lu_reg/{i:05d}.pt")
        end_t = time()

        if i % 50 == 0:
            print(f"[{i:05d}/{N_CUTOFF}] in {end_t - start_t:03f}s")
            vis("tmp/better_remap.png", pil_img, lr_feats, hr_feats, None, mlp)
        if i == N_CUTOFF:
            break

# img = Image.open(image_path).convert("RGB")
# normalized_img_tensor = transform(img).unsqueeze(0).to("cuda")
# lr_feats = model(normalized_img_tensor)  # 1, dim, lr_size, lr_size

# ## Upsampling step
# hr_feats = upsampler(lr_feats, normalized_img_tensor)  # 1, dim, 224, 224

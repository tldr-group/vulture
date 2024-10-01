import torch
import torchvision.transforms as T
import argparse
import hydra
from omegaconf import OmegaConf
from os import listdir
from PIL import Image
from featup.train_implicit_upsampler import my_app

from utils import do_2D_pca, to_numpy, do_pca

torch.manual_seed(1001)

DEVICE = "cuda:0"
dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# dv2 = add_flash_attention(dv2)
dv2 = dv2.eval().to(DEVICE)  # .half()

with open("FeatUp/featup/configs/implicit_upsampler.yaml") as f:
    basic_conf_file = f.read()
basic_conf = OmegaConf.create(basic_conf_file)


@hydra.main(config_path="", config_name="")
def start_featup(split_n: int) -> None:
    torch.set_num_threads(20)
    new_config = basic_conf.copy()
    new_config.dataset = f"data/imagenet_reduced/splits/{split_n}"
    print(new_config)
    my_app(new_config)


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr = T.Compose(
    [
        T.ToTensor(),
        norm,
    ]
)


@torch.no_grad()
def gen_original_dv2_embeds(model: torch.nn.Module, folder: str) -> None:
    data_path = "data/imagenet_reduced"
    for i, fname in enumerate(listdir(f"{data_path}/{folder}")):
        img_idx = int(fname.split(".")[0])
        img_fname = f"{data_path}/splits/{img_idx % 10}/{img_idx}.png"
        img = Image.open(img_fname).convert("RGB")

        tensor: torch.Tensor = tr(img).unsqueeze(0).cuda()
        b, _, h, w = tensor.shape
        nt_w: int = 1 + (w - 14) // 14
        nt_h: int = 1 + (h - 14) // 14
        feat_dict: dict = model.forward_features(tensor)
        feats = feat_dict["x_norm_patchtokens"]
        _, nt, c = feats.shape
        # feats = feats.permute((0, 2, 1))

        feats_np = to_numpy(feats.detach())[0]
        reduced = do_pca(feats_np, 128)

        # feats = feats.reshape((b, c, nt_h, nt_w)).detach().cpu()

        # reduced = do_2D_pca(to_numpy(feats), 128)
        reduced_tensor = torch.tensor(reduced).T.reshape((1, 128, nt_h, nt_w))

        data = torch.load(f"{data_path}/{folder}/{fname}", weights_only=True)
        data["dv2_lr_feats_reduced"] = reduced_tensor
        torch.save(data, f"{data_path}/{folder}/{fname}")

        print(i)


parser = argparse.ArgumentParser(description="get split")
parser.add_argument(
    "--n",
    type=int,
    default=0,
    metavar="N",
)
if __name__ == "__main__":
    args = parser.parse_args()
    # so far have done 0 and 2
    # need to do 3, 4, 5, 6, 7, 8, 9
    start_featup(args.n)
    # gen_original_dv2_embeds(dv2, "data")

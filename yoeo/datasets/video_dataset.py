import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TF  # type: ignore
import numpy as np
from scipy.stats import truncnorm
from PIL import Image
from os import listdir

from yoeo.datasets.lr_hr_embedding_dataset import unnorm
from yoeo.utils import (
    Experiment,
    paired_frames_vis,
    propagator_batch_vis,
    expriment_from_json,
    get_shortest_side_resize_dims,
    resize_crop,
)

from timm.models import VisionTransformer
from typing import Literal

Datasets = Literal["lvos", "mose"]
Splits = Literal["train", "val"]

MIN_L = 518


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        datasets: list[Datasets],
        which: Literal[Splits],
        expr: Experiment,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.datasets = datasets
        self.subdir = which
        self.expr = expr

        self.map = self.get_map(self.root_dir, self.datasets, self.subdir)
        self.folder_names = list(self.map.keys())
        self.n = len(self.folder_names)
        print(f"{self.n} files for '{self.subdir}' for {self.datasets}")

    def _get_files_or_folders_at_path(
        self, root: str, ds: Datasets, split: Splits, vid_name: str | None
    ) -> list[str]:
        if vid_name is not None:
            all_files = listdir(f"{root}/{ds}/{split}/JPEGImages/{vid_name}")
        else:
            all_files = listdir(f"{root}/{ds}/{split}/JPEGImages")
        filtered = [
            f for f in all_files if f[0] != "."
        ]  # mose has duplicate/checksum folders/files with ._ prefixs
        return filtered

    def get_map(
        self, root: str, datasets: list[Datasets], split: Splits
    ) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for ds in datasets:
            folders = self._get_files_or_folders_at_path(root, ds, split, None)
            for folder in folders:
                files = self._get_files_or_folders_at_path(root, ds, split, folder)
                out[f"{root}/{ds}/{split}/JPEGImages/{folder}"] = files
        return out

    def _get_frames(self, frames: list[str]) -> list[str]:
        inds = np.arange(len(frames))
        N = len(inds)
        first_frame_idx = np.random.choice(inds)

        loc = first_frame_idx / float(N)
        scale = 0.05
        a_trunc, b_trunc = 0, 1
        a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
        rv = truncnorm(a, b, loc=loc, scale=scale)
        r = rv.rvs(size=1)
        second_frame_idx = int(np.floor(r * (N - 1)))
        return [frames[first_frame_idx], frames[second_frame_idx]]

    def _get_dir(self, folder: str, frame: str) -> str:
        return f"{self.root_dir}/{self.subdir}/JPEGImages/{folder}/{frame}.jpg"

    def __len__(self):
        return self.n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        folder = self.folder_names[index]
        frames = self.map[folder]
        chosen_frames = self._get_frames(frames)

        imgs: list[torch.Tensor] = []
        for i in range(2):
            pil = Image.open(f"{folder}/{chosen_frames[i]}")
            tensor = TF.pil_to_tensor(pil)
            tensor = tensor.to(torch.float32)
            h, w = get_shortest_side_resize_dims(pil.height, pil.width, MIN_L)
            resized = TF.resize(tensor, [h, w])
            cropped = TF.center_crop(resized, [518, 518])
            normed = TF.normalize(cropped, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            imgs.append(normed)
        imgs_0, imgs_1 = imgs
        return (imgs_0, imgs_1)

    @torch.no_grad()
    def get_features_of_batches(
        self, model: VisionTransformer, img_0: torch.Tensor, img_1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = img_0.shape
        n_patch_h, n_patch_w = h // self.expr.patch_size, w // self.expr.patch_size

        input_feat_dict: dict[str, torch.Tensor] = model.forward_features(img_0)  # type: ignore
        target_feat_dict: dict[str, torch.Tensor] = model.forward_features(img_1)  # type: ignore

        flat_input_feats = input_feat_dict["x_norm_patchtokens"]
        flat_target_feats = target_feat_dict["x_norm_patchtokens"]
        b, _, c = flat_input_feats.shape

        if self.expr.norm:  # normalize along channel dims
            flat_input_feats = F.normalize(flat_input_feats, p=1, dim=-1)
            flat_target_feats = F.normalize(flat_target_feats, p=1, dim=-1)
        # we want features in shape (B,C,H,W) for network
        flat_input_feats = flat_input_feats.permute((0, 2, 1))
        flat_target_feats = flat_target_feats.permute((0, 2, 1))

        input_feats = flat_input_feats.reshape((b, c, n_patch_h, n_patch_w))
        target_feats = flat_target_feats.reshape((b, c, n_patch_h, n_patch_w))

        return (input_feats, target_feats)


DEVICE = "cuda:1"
if __name__ == "__main__":

    dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    dv2 = dv2.eval().to(DEVICE)

    expr = expriment_from_json("yoeo/models/configs/simple_no_trs_dv2.json")
    ds = VideoDataset("data", ["lvos", "mose"], "train", expr)
    dl = DataLoader(ds, 20, True)
    img_0, img_1 = next(iter(dl))
    img_0 = img_0.to(DEVICE)
    img_1 = img_1.to(DEVICE)

    inp_feats, outp_feats = ds.get_features_of_batches(dv2, img_0, img_1)

    img_0 = unnorm(img_0.to("cpu")).to(torch.uint8)
    img_1 = unnorm(img_1.to("cpu")).to(torch.uint8)

    n_samples = 8
    # paired_frames_vis(img_0[:n_samples], img_1[:n_samples], "out.png")
    propagator_batch_vis(
        img_0, img_1, inp_feats, outp_feats, outp_feats, "prop_batch_vis.png"
    )

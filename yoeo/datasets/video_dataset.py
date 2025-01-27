import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TF  # type: ignore
from json import load
import numpy as np
from PIL import Image

from yoeo.utils import Experiment, paired_frames_vis, expriment_from_json

from timm.models import VisionTransformer
from typing import Literal


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        which: Literal["train", "val"],
        expr: Experiment,
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.subdir = which
        self.expr = expr

        json = self.get_meta_json()
        self.map: dict[str, list[str]] = self.get_map_from_json(json)
        self.folder_names = list(self.map.keys())
        self.n = len(self.folder_names)

    def get_meta_json(self) -> dict:
        with open(f"{self.root_dir}/{self.subdir}/meta.json") as f:
            json = load(f)
        return json["videos"]  # file json nested under 'json'

    def get_map_from_json(self, json: dict) -> dict[str, list[str]]:
        # go from json -> dict of fnames <-> frames in that fname
        out_map: dict[str, list[str]] = {}
        for video_folder, child_dict in json.items():
            frame_range = child_dict["objects"]["1"]["frame_range"]
            # frames aren't 000001, 000002 etc but are 00001, 00006, 00011 etc.
            start = int(frame_range["start"])
            end = int(frame_range["end"])
            N = int(frame_range["frame_nums"])
            inc = int((end - start) / N)
            frames = [f"{(start + i * inc):08d}" for i in range(N)]
            out_map[video_folder] = frames
        return out_map

    def _get_dir(self, folder: str, frame: str) -> str:
        return f"{self.root_dir}/{self.subdir}/JPEGImages/{folder}/{frame}.jpg"

    def __len__(self):
        return self.n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        folder = self.folder_names[index]
        frames = self.map[folder]
        chosen_frames = list(np.random.choice(frames, 2, replace=False))

        imgs: list[torch.Tensor] = []
        for i in range(2):
            pil = Image.open(self._get_dir(folder, chosen_frames[i]))
            tensor = TF.pil_to_tensor(pil)
            tensor = tensor.to(torch.float32)
            cropped = TF.center_crop(tensor, [518, 518])
            normed = TF.normalize(cropped, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            imgs.append(normed)
        imgs_0, imgs_1 = imgs
        return (imgs_0, imgs_1)

    @torch.no_grad()
    def get_features_of_batches(
        self, model: VisionTransformer, imgs_0: torch.Tensor, imgs_1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = img_0.shape
        n_patch_h, n_patch_w = h // self.expr.patch_size, w // self.expr.patch_size

        input_feat_dict: dict[str, torch.Tensor] = model.forward_features(imgs_0)  # type: ignore
        target_feat_dict: dict[str, torch.Tensor] = model.forward_features(imgs_1)  # type: ignore

        flat_input_feats = input_feat_dict["x_norm_patchtokens"]
        flat_target_feats = input_feat_dict["x_norm_patchtokens"]
        b, n_t, c = flat_input_feats.shape

        if self.expr.norm:
            flat_input_feats = F.normalize(flat_input_feats, p=1, dim=1)
            flat_target_feats = F.normalize(flat_input_feats, p=1, dim=1)

        input_feats = flat_input_feats.reshape((b, n_patch_h, n_patch_w, c))
        target_feats = flat_target_feats.reshape((b, n_patch_h, n_patch_w, c))

        return (input_feats, target_feats)


DEVICE = "cuda:1"
if __name__ == "__main__":

    dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    dv2 = dv2.eval().to(DEVICE)

    expr = expriment_from_json("yoeo/models/configs/simple_no_trs_dv2.json")
    ds = VideoDataset("data/lvos", "train", expr)
    dl = DataLoader(ds, 20, True)
    img_0, img_1 = next(iter(dl))
    img_0 = img_0.to(DEVICE)
    img_1 = img_1.to(DEVICE)

    inp_feats, outp_feats = ds.get_features_of_batches(dv2, img_0, img_1)
    print(inp_feats.shape)

    img_0 = img_0.to("cpu")
    img_1 = img_1.to("cpu")

    n_samples = 8
    paired_frames_vis(img_0[:n_samples], img_1[:n_samples], "out.png")

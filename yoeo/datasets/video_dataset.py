import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF  # type: ignore
from json import load
import numpy as np
from PIL import Image

from yoeo.utils import Experiment, paired_frames_vis

from typing import Literal


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        which: Literal["train", "val"],
        expr: Experiment | None,
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
        return (imgs[0], imgs[1])


if __name__ == "__main__":
    ds = VideoDataset("data/lvos", "train", None)
    dl = DataLoader(ds, 20, True)
    img_0, img_1 = next(iter(dl))

    n_samples = 8
    paired_frames_vis(img_0[:n_samples], img_1[:n_samples], "out.png")

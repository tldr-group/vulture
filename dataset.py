import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from os import listdir
from PIL import Image
from typing import Literal

from utils import visualise, Experiment


import warnings

warnings.filterwarnings("ignore")

# transforms: flips (ud, lr, ud-lr), rotations (90, 180, 270)


def unnorm(x: torch.Tensor) -> torch.Tensor:
    return TF.normalize(x, [-0.485, -0.456, -0.406], [1 / 0.229, 1 / 0.224, 1 / 0.225])


def shift(x: torch.Tensor, s: int, dir: tuple[int, int]) -> torch.Tensor:
    return torch.roll(x, (dir[0] * s, dir[1] * s), dims=(-2, -1))


class EmbeddingDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        which: Literal["train", "val"],
        expr: Experiment,
        using_splits: bool = True,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.subdir = "data" if which == "train" else which
        self.files = sorted(
            [
                f"{root_dir}/{self.subdir}/{p}"
                for p in listdir(f"{root_dir}/{self.subdir}")
            ]
        )
        self.n = len(self.files)
        self.device = device

        self.img_dir = (
            f"{self.root_dir}/splits" if using_splits else f"{self.root_dir}/imgs"
        )
        self.expr = expr

    def __len__(self):
        return self.n

    def transform(
        self, img: Image.Image, lr_feats: torch.Tensor, hr_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img_tensor = TF.pil_to_tensor(img)
        flip_h: bool = torch.rand(1) < self.expr.flip_h_prob  # type: ignore
        flip_v: bool = torch.rand(1) < self.expr.flip_v_prob  # type: ignore

        angles_deg, shift_dirs, shift_dists = (
            self.expr.angles_deg,
            self.expr.shift_dirs,
            self.expr.shift_dists,
        )

        if flip_h:
            img_tensor = TF.hflip(img_tensor)
            lr_feats = TF.hflip(lr_feats)
            hr_feats = TF.hflip(hr_feats)
        if flip_v:
            img_tensor = TF.vflip(img_tensor)
            lr_feats = TF.vflip(lr_feats)
            hr_feats = TF.vflip(hr_feats)

        if len(angles_deg) > 0:
            rotate_deg = angles_deg[torch.randint(len(angles_deg), (1,))]
            img_tensor = TF.rotate(img_tensor, rotate_deg)
            lr_feats = TF.rotate(lr_feats, rotate_deg)
            hr_feats = TF.rotate(hr_feats, rotate_deg)

        if len(shift_dirs) > 0 and len(shift_dists) > 0:
            shift_dir = shift_dirs[torch.randint(len(shift_dirs), (1,))]
            shift_dist = shift_dists[torch.randint(len(shift_dists), (1,))]
            img_tensor = shift(img_tensor, shift_dist, shift_dir)
            hr_feats = shift(hr_feats, shift_dist, shift_dir)

        if self.expr.norm:
            img_tensor = img_tensor.to(torch.float32)
            img_tensor = TF.normalize(
                img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            lr_feats = F.normalize(lr_feats, p=1, dim=0)
            hr_feats = F.normalize(hr_feats, p=1, dim=0)
        return img_tensor, lr_feats, hr_feats

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_file = self.files[index]
        chosen_fname_val = int(chosen_file.split("/")[-1].split(".")[0])
        # load directly to device we specify - usually gpu
        embedding_data = torch.load(
            chosen_file,
            map_location=self.device,
            weights_only=True,
        )
        # dataloader will batch for us

        if self.expr.net_type != "transfer":
            lr_feats, hr_feats = (
                embedding_data["dv2_lr_feats_reduced"][0],
                embedding_data["hr_feats"][0],
            )
        else:
            lr_feats, hr_feats = (
                embedding_data["dv2_lr_feats"][0],
                embedding_data["lr_feats"][0],
            )

        img = Image.open(
            f"{self.img_dir}/{chosen_fname_val % 10}/{chosen_fname_val}.png"
        )
        # randomly sample transform, apply to img, lr, hr
        return self.transform(img, lr_feats, hr_feats)


if __name__ == "__main__":
    ds = EmbeddingDataset(
        "data/imagenet_reduced", "val", Experiment("test", "transfer")
    )
    dl = DataLoader(ds, 20, True)
    img, lr, hr = next(iter(dl))
    print(img.shape, lr.shape, hr.shape)
    visualise(unnorm(img).to(torch.uint8), lr, hr, hr, "batch_vis.png")


""""


FLIP_H_PROB, FLIP_V_PROB = 0.5, 0.5
ANGLES_DEG = [0, 0, 0, 0, 90, 180, 270]
SHIFT_DIRS = [
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]
SHIFT_DISTS = [0, 0, 1, 2, 3, 4]
"""

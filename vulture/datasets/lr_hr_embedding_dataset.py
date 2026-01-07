import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TF  # type: ignore
from os import listdir
from PIL import Image
from typing import Literal

# from vulture.datasets.learn_remap_LU_feats import apply
from vulture.utils import visualise, Experiment


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
        device: str = "cuda:1",
        data_suffix: Literal["", "_reg", "_fit_reg", "_lu_reg", "_jf_reg"] = "",
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.subdir = "data" + data_suffix if which == "train" else which + data_suffix
        self.files = sorted([f"{root_dir}/{self.subdir}/{p}" for p in listdir(f"{root_dir}/{self.subdir}")])

        self.n = len(self.files)
        self.device = device

        self.using_splits = using_splits
        self.img_dir = f"{self.root_dir}/splits" if using_splits else f"{self.root_dir}/imgs"
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

        img_tensor = img_tensor.to(torch.float32)
        img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.expr.load_size != 224:
            img_tensor = TF.resize(img_tensor, [self.expr.load_size, self.expr.load_size])

        if self.expr.norm:
            lr_feats = F.normalize(lr_feats, p=1, dim=0)
            hr_feats = F.normalize(hr_feats, p=1, dim=0)
        return img_tensor, lr_feats, hr_feats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen_file = self.files[index]
        chosen_fname = str(chosen_file.split("/")[-1].split(".")[0])
        chosen_fname_val = int(chosen_fname)
        # load directly to device we specify - usually gpu
        embedding_data = torch.load(
            chosen_file,
            map_location=self.device,
            weights_only=True,
        )
        # dataloader will batch for us

        if self.expr.net_type != "transfer":
            lr_feats, hr_feats = (
                embedding_data["lr_feats"][0],
                embedding_data["hr_feats"][0],
            )
        else:
            lr_feats, hr_feats = (
                embedding_data["dv2_lr_feats"][0],
                embedding_data["lr_feats"][0],
            )

        if self.using_splits:
            img = Image.open(f"{self.img_dir}/{chosen_fname_val % 10}/{chosen_fname_val}.png")
        else:
            img = Image.open(f"{self.img_dir}/{chosen_fname}.png")

        # if self.apply_mlp:
        #     weights = embedding_data["mlp_weights"]
        #     self.mlp.load_state_dict(weights)
        #     hr_feats = apply(self.mlp, hr_feats.unsqueeze(0))
        #     hr_feats = hr_feats[0]

        if lr_feats.shape[0] != self.expr.n_ch_in:
            lr_feats = lr_feats[: self.expr.n_ch_in]
        if hr_feats.shape[0] != self.expr.n_ch_out:
            hr_feats = hr_feats[: self.expr.n_ch_out]
        # randomly sample transform, apply to img, lr, hr
        return self.transform(img, lr_feats, hr_feats)


if __name__ == "__main__":
    ds = EmbeddingDataset(
        "data/imagenet_reduced",
        "train",
        Experiment("test", n_ch_in=48, n_ch_out=48, norm=True),
        data_suffix="_jf_reg",
        using_splits=False,
        device="cuda:1",
    )
    dl = DataLoader(ds, 20, True)
    next(iter(dl))
    img, lr, hr = next(iter(dl))
    print(img.shape, lr.shape, hr.shape)
    visualise(unnorm(img).to(torch.uint8), lr, hr, hr, "tmp/batch_vis.png", False)


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

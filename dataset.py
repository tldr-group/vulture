import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from os import listdir
from PIL import Image
from typing import Literal

from utils import visualise


import warnings

warnings.filterwarnings("ignore")

# transforms: flips (ud, lr, ud-lr), rotations (90, 180, 270)


FLIP_H_PROB, FLIP_V_PROB = 0.3, 0.3
ANGLES_DEG = [0, 0, 0, 0, 90, 180, 270]


class EmbeddingDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        which: Literal["train", "val"],
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

    def __len__(self):
        return self.n

    def transform(
        self, img: Image.Image, lr_feats: torch.Tensor, hr_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img_tensor = TF.pil_to_tensor(img)
        flip_h: bool = torch.rand(1) > FLIP_H_PROB  # type: ignore
        flip_v: bool = torch.rand(1) > FLIP_V_PROB  # type: ignore
        rotate_deg = ANGLES_DEG[torch.randint(len(ANGLES_DEG), (1,))]
        for tensor in (img_tensor, lr_feats, hr_feats):
            if flip_h:
                tensor = TF.hflip(tensor)
            if flip_v:
                tensor = TF.vflip(tensor)
            tensor = TF.rotate(tensor, rotate_deg)
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
        lr_feats, hr_feats = (
            embedding_data["lr_feats"][0],
            embedding_data["hr_feats"][0],
        )

        img = Image.open(
            f"{self.img_dir}/{chosen_fname_val % 10}/{chosen_fname_val}.png"
        )
        # randomly sample transform, apply to img, lr, hr
        return self.transform(img, lr_feats, hr_feats)


if __name__ == "__main__":
    ds = EmbeddingDataset("data/imagenet_reduced", "train")
    dl = DataLoader(ds, 10, True)
    img, lr, hr = next(iter(dl))
    print(img.shape, lr.shape, hr.shape)
    visualise(img, lr, hr, "batch_vis.png")

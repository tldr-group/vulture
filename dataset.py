from os import listdir
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from utils import visualise
import warnings

warnings.filterwarnings("ignore")

# transforms: flips (ud, lr, ud-lr), rotations (90, 180, 270)


class EmbeddingDataset(Dataset):

    def __init__(
        self, root_dir: str, using_splits: bool = True, device: str = "cuda:0"
    ) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.files = sorted(
            [f"{root_dir}/data/{p}" for p in listdir(f"{root_dir}/data")]
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
        flip_h: bool = torch.rand(1) > 0.3  # type: ignore
        flip_v: bool = torch.rand(1) > 0.3  # type: ignore
        angles = [0, 0, 0, 0, 90, 180, 270]
        rotate_deg = angles[torch.randint(len(angles), (1,))]
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

        embedding_data = torch.load(
            chosen_file,
            map_location=self.device,
            weights_only=True,
        )
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
    ds = EmbeddingDataset("data/imagenet_reduced")
    dl = DataLoader(ds, 10, True)
    img, lr, hr = next(iter(dl))
    print(img.shape, lr.shape, hr.shape)
    visualise(img, lr, hr, "batch_vis.png")

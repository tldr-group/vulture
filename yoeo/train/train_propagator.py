import torch
import torch.nn as nn

import numpy as np

torch.manual_seed(0)
np.random.seed(0)

from yoeo.datasets import VideoDataset, DataLoader
from yoeo.models import FeaturePropagator
from yoeo.utils import expriment_from_json, init_weights, add_flash_attention


torch.cuda.empty_cache()
DEVICE = "cuda:1"


dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
dv2 = add_flash_attention(dv2)
dv2 = dv2.eval().to(DEVICE).half()


expr = expriment_from_json("yoeo/models/configs/simple_no_trs_dv2.json")
print(expr)

train_ds = VideoDataset("data/lvos", "train", expr=expr)
train_dl = DataLoader(train_ds, expr.batch_size, True)

net = FeaturePropagator(
    expr.patch_size,
    n_ch_imgs=expr.n_ch_guidance,
    n_ch_in=expr.n_ch_in,
    n_ch_out=expr.n_ch_out,
    k=expr.k,
).to(DEVICE)

init_weights(net, expr.weights_init)


opt_dict = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
opt: torch.optim.Optimizer = opt_dict[expr.optim](net.parameters(), lr=expr.lr)
N_EPOCHS = expr.n_epochs
SAVE_PER = expr.save_per

loss_dict: dict = {"smooth_l1": nn.SmoothL1Loss, "l1": nn.L1Loss, "l2": nn.MSELoss}
loss_fn: nn.modules.loss._Loss = loss_dict[expr.loss](reduction="sum")


def feed_batch_get_loss(
    model: nn.Module,
    opt,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    training: bool = True,
) -> float:
    if training:
        model.train()
        opt.zero_grad()
    else:
        model.eval()

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

torch.manual_seed(0)
np.random.seed(0)

from yoeo.datasets import EmbeddingDataset, DataLoader, unnorm
from yoeo.models import FeatureUpsampler
from yoeo.utils import visualise, plot_losses, expriment_from_json, init_weights


torch.cuda.empty_cache()


DEVICE = "cuda:1"

expr = expriment_from_json("yoeo/models/configs/combined_no_shift.json")
print(expr)

train_ds = EmbeddingDataset(
    "data/imagenet_reduced", "train", expr=expr, device=DEVICE, data_suffix="_reg"
)
val_ds = EmbeddingDataset(
    "data/imagenet_reduced", "val", expr=expr, device=DEVICE, data_suffix="_reg"
)

train_dl = DataLoader(train_ds, expr.batch_size, True)
val_dl = DataLoader(
    val_ds,
    expr.batch_size,
    True,
)

net = FeatureUpsampler(
    expr.patch_size, k_up=expr.k, n_ch_in=expr.n_ch_in, feat_weight=expr.feat_weight
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

scheduler = ReduceLROnPlateau(opt, patience=20)
# loss_fn = torch.nn.MSELoss(reduction="sum")


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

    img, lr_feats, hr_feats = batch
    img, lr_feats, hr_feats = (
        img.to(DEVICE).to(torch.float32),
        lr_feats.to(DEVICE),
        hr_feats.to(DEVICE),
    )

    pred_hr_feats = model(img, lr_feats)
    loss = loss_fn(pred_hr_feats, hr_feats)
    if training:
        loss.backward()
        opt.step()

    img, lr_feats, hr_feats = (
        img.to("cpu"),
        lr_feats.to("cpu"),
        hr_feats.to("cpu"),
    )

    return loss.item()


train_losses, val_losses = [], []
best_val_loss = 1e10
for i in range(N_EPOCHS):
    epoch_loss = 0.0
    for batch in train_dl:
        loss_val = feed_batch_get_loss(net, opt, batch)
        epoch_loss += loss_val

    val_loss = 0.0
    for batch in val_dl:
        val_loss += feed_batch_get_loss(net, opt, next(iter(val_dl)), False)
    print(f"[{i}/{N_EPOCHS}]: train={epoch_loss}, val={val_loss}")
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    if i % SAVE_PER == 0:
        img, lr_feats, hr_feats = next(iter(val_dl))
        img, lr_feats, hr_feats = (
            img.to(DEVICE).to(torch.float32),
            lr_feats.to(DEVICE),
            hr_feats.to(DEVICE),
        )

        pred_hr_feats = net(img, lr_feats)
        img = unnorm(img)
        img = img.to(torch.uint8)
        visualise(
            img, lr_feats, hr_feats, pred_hr_feats, f"experiments/current/val_{i}.png"
        )
        plot_losses(train_losses, val_losses, f"experiments/current/losses.png")

        if val_loss < best_val_loss:
            # todo: just save every 100 epochs?
            torch.save(net.state_dict(), f"experiments/current/best.pth")
            best_val_loss = val_loss

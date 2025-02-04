import torch
import torch.nn as nn

import numpy as np

torch.manual_seed(0)
np.random.seed(0)

from yoeo.datasets import VideoDataset, DataLoader, unnorm
from yoeo.models import FeaturePropagator
from yoeo.utils import (
    expriment_from_json,
    init_weights,
    add_flash_attention,
    propagator_batch_vis,
    plot_losses,
)


torch.cuda.empty_cache()
DEVICE = "cuda:1"


dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
dv2 = add_flash_attention(dv2)
dv2 = dv2.eval().to(DEVICE).half()


expr = expriment_from_json("yoeo/models/configs/simple_prop.json")
print(expr)

train_ds = VideoDataset("data", ["lvos", "mose"], "train", expr=expr)
train_dl = DataLoader(train_ds, expr.batch_size, True)

val_ds = VideoDataset("data", ["lvos", "mose"], "val", expr=expr)
val_dl = DataLoader(train_ds, expr.batch_size, True)

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
    model: FeaturePropagator,
    ref_feat_model: nn.Module,
    opt,
    img_batch: tuple[torch.Tensor, torch.Tensor],
    training: bool = True,
) -> float:
    if training:
        model.train()
        opt.zero_grad()
    else:
        model.eval()

    img_0, img_1 = img_batch

    img_0, img_1 = (
        img_0.to(DEVICE).to(torch.float32),
        img_1.to(DEVICE).to(torch.float32),
    )

    input_feats, target_feats = train_ds.get_featup_features_of_batches(
        ref_feat_model, img_0, img_1
    )

    input_feats, target_feats = (
        input_feats.to(DEVICE),
        target_feats.to(DEVICE),
    )

    pred_feats = model.forward(input_feats, img_0, img_1)

    loss = loss_fn(pred_feats, target_feats)
    if training:
        loss.backward()
        opt.step()

    img_0, img_1, input_feats, target_feats = (
        img_0.to("cpu"),
        img_1.to("cpu"),
        input_feats.to("cpu"),
        target_feats.to("cpu"),
    )

    return loss.item()


train_losses, val_losses = [], []
best_val_loss = 1e10
for i in range(N_EPOCHS):
    epoch_loss = 0.0
    for batch in train_dl:
        loss_val = feed_batch_get_loss(net, dv2, opt, batch)
        epoch_loss += loss_val

    val_loss = feed_batch_get_loss(net, dv2, opt, next(iter(val_dl)), False)
    print(f"[{i}/{N_EPOCHS}]: train={epoch_loss}, val={val_loss}")
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    if i % SAVE_PER == 0:
        img_0, img_1 = next(iter(val_dl))

        img_0, img_1 = (
            img_0.to(DEVICE).to(torch.float32),
            img_1.to(DEVICE).to(torch.float32),
        )

        input_feats, target_feats = train_ds.get_featup_features_of_batches(
            dv2, img_0, img_1
        )

        input_feats, target_feats = (
            input_feats.to(DEVICE),
            target_feats.to(DEVICE),
        )

        pred_feats = net.forward(input_feats, img_0, img_1)

        img_0 = unnorm(img_0.to("cpu")).to(torch.uint8)
        img_1 = unnorm(img_1.to("cpu")).to(torch.uint8)

        propagator_batch_vis(
            img_0,
            img_1,
            input_feats,
            target_feats,
            pred_feats,
            f"experiments/current/val_{i}.png",
            True,
        )

        plot_losses(train_losses, val_losses, f"experiments/current/losses.png")

        if val_loss < best_val_loss:
            # todo: just save every 100 epochs?
            torch.save(net.state_dict(), f"experiments/current/best.pth")
            best_val_loss = val_loss

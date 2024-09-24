import torch
import torch.nn as nn

from dataset import EmbeddingDataset, DataLoader
from model import Combined
from utils import visualise

# torch.cuda.empty_cache()

DEVICE = "cuda:1"

train_ds = EmbeddingDataset("data/imagenet_reduced", "train", device=DEVICE)
val_ds = EmbeddingDataset("data/imagenet_reduced", "val", device=DEVICE)

train_dl = DataLoader(train_ds, 40, True)
val_dl = DataLoader(val_ds, 20, True)

net = Combined(14).to(DEVICE)

opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
N_EPOCHS = 1000
SAVE_PER = 1

loss_fn = torch.nn.MSELoss()


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
    return loss.item()


for i in range(N_EPOCHS):
    epoch_loss = 0.0
    for batch in train_dl:
        loss_val = feed_batch_get_loss(net, opt, batch)
        epoch_loss += loss_val

    val_loss = feed_batch_get_loss(net, opt, next(iter(val_dl)), False)
    print(f"[{i}/{N_EPOCHS}]: train={epoch_loss:.4f}, val={val_loss:.4f}")

    if i % SAVE_PER == 0:
        img, lr_feats, hr_feats = next(iter(val_dl))
        img, lr_feats, hr_feats = (
            img.to(DEVICE).to(torch.float32),
            lr_feats.to(DEVICE),
            hr_feats.to(DEVICE),
        )

        pred_hr_feats = net(img, lr_feats)

        visualise(img, lr_feats, hr_feats, pred_hr_feats, f"experiments/val_{i}.png")

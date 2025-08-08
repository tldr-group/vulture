import torch
import random

from os import getcwd
import numpy as np
import matplotlib.pyplot as plt
import pretrained_microscopy_models as pmm

import yoeo.comparisons.pmm_helpers as pmm_h

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

CWD = getcwd()

device = "cuda"
arch: str = "UnetPlusPlus"
encoder: str = "resnet50"
pretrained_weights: str = "micronet"
sparse: bool = False

model = pmm_h.get_model(n_classes=3, arch=arch, encoder=encoder, pretrained_weights=pretrained_weights)

class_values = {
    "matrix": [85, 85, 85],
    "secondary": [170, 170, 170],
    "tertiary": [255, 255, 255],
}
class_values_original = {"matrix": [0, 0, 0], "secondary": [255, 0, 0], "tertiary": [0, 0, 255]}
class_values_plus_unlabelled = {
    "matrix": [85, 85, 85],
    "secondary": [170, 170, 170],
    "tertiary": [255, 255, 255],
    "unlabelled": [0, 0, 0],
}

values = class_values_plus_unlabelled if sparse else class_values

DATA_PATH = f"{CWD}/paper_figures/fig_data/CNN_comparison/ni_superalloy"
train_path = f"{DATA_PATH}/train_sparse_annot" if sparse else f"{DATA_PATH}/train_annot"
train_ds = pmm_h.get_dataset(values, f"{DATA_PATH}/train", train_path, "train")
val_ds = pmm_h.get_dataset(values, f"{DATA_PATH}/val", f"{DATA_PATH}/val_annot", "val")
test_ds = pmm_h.get_dataset(values, f"{DATA_PATH}/test", f"{DATA_PATH}/test_annot", "val")

masked_loss = pmm_h.MaskedDiceBCELoss(weight=0.7)
default_loss = pmm.losses.DiceBCELoss(weight=0.7)

loss = masked_loss if sparse else default_loss

state = pmm.segmentation_training.train_segmentation_model(
    model=model,
    loss=loss,
    architecture=arch,
    encoder=encoder,
    train_dataset=train_ds,
    validation_dataset=val_ds,
    class_values=class_values,
    patience=30,
    device=device,
    lr=2e-4,
    batch_size=6,
    val_batch_size=6,
    save_folder=f"{CWD}/paper_figures/models",
    epochs=100,
    save_name="foo",
)

plt.plot(state["train_loss"], label="train_loss")
plt.plot(state["valid_loss"], label="valid_loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss_full.png")

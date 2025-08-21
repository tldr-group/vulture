import torch
import random

import numpy as np
import matplotlib.pyplot as plt
import pretrained_microscopy_models as pmm

import vulture.comparisons.pmm_helpers as pmm_h

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = "cuda"
arch: str = "UnetPlusPlus"
encoder: str = "resnet50"
pretrained_weights: str = "micronet"
sparse: bool = False


n_epochs = 34
# for full run needs to be 200
save_per = 1
N_REPEATS = 1

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


FIG_PATH = "paper_figures/"
DATA_PATH = f"{FIG_PATH}/fig_data/CNN_comparison/ni_superalloy"
train_path = f"{DATA_PATH}/train_sparse_annot" if sparse else f"{DATA_PATH}/train_annot"
train_ds = pmm_h.get_dataset(values, f"{DATA_PATH}/train", train_path, "train")
val_ds = pmm_h.get_dataset(values, f"{DATA_PATH}/val", f"{DATA_PATH}/val_annot", "val")
test_ds = pmm_h.get_dataset(values, f"{DATA_PATH}/test", f"{DATA_PATH}/test_annot", "val")

masked_loss = pmm_h.MaskedDiceBCELoss(weight=0.7)
default_loss = pmm.losses.DiceBCELoss(weight=0.7)

loss = masked_loss if sparse else default_loss


print(f"{n_epochs} epochs, save {save_per}, {N_REPEATS} repeats, sparse: {sparse}")

results = []
for i in range(N_REPEATS):
    model = pmm_h.get_model(n_classes=3, arch=arch, encoder=encoder, pretrained_weights=pretrained_weights)
    results_dict_list, state = pmm_h.train_segmentation_model_with_eval(
        model, arch, encoder, train_ds, val_ds, values, 3, loss, n_epochs, save_per
    )
    if i > 1:
        for results_dict in results_dict_list:
            # delete these to save space
            results_dict["gts"] = []
            results_dict["preds"] = []

    results.append(results_dict_list)
    print(f"Training run for repeat {i} finished!")

out_dir = f"{FIG_PATH}/fig_data/CNN_comparison/ni_superalloy/stored_CNN_results/"
prefix = "sparse" if sparse else "full"
out_path = f"{out_dir}/{prefix}_4_imgs_e{n_epochs}_avg{N_REPEATS}.npy"
np.save(out_path, results)

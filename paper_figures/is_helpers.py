import torch
import numpy as np
from PIL import Image

from yoeo.main import (
    get_hr_feats,
    get_lr_feats,
    convert_image,
    closest_crop,
    Experiment,
)
from yoeo.models import FeatureUpsampler
from yoeo.utils import to_numpy

from os import listdir

from interactive_seg_backend import featurise_
from interactive_seg_backend.configs import TrainingConfig
from interactive_seg_backend.classifiers.base import Classifier
from interactive_seg_backend.file_handling import (
    load_labels,
    load_image,
)
from interactive_seg_backend.core import (
    train,
    get_training_data,
    shuffle_sample_training_data,
    get_model,
)
from interactive_seg_backend.main import apply
from interactive_seg_backend.utils import class_avg_miou


from typing import Literal


DEVICE = "cuda:1"

AllowedDatasets = Literal["Cu_ore_RLM", "Ni_superalloy_SEM", "T_cell_TEM"]


def get_deep_feats(
    img: Image.Image,
    dv2: torch.nn.Module,
    upsampler: FeatureUpsampler,
    expr: Experiment,
    K: int = 32,
    existing_pca: object | None = None,
) -> np.ndarray:
    hr_feats = get_hr_feats(img, dv2, upsampler, DEVICE, n_ch_in=expr.n_ch_in, existing_pca=existing_pca)
    hr_feats_np = to_numpy(hr_feats)
    hr_feats_np = hr_feats_np.transpose((1, 2, 0))[:, :, :K]
    return hr_feats_np


def train_model_over_images(
    dataset: AllowedDatasets,
    train_cfg: TrainingConfig,
    path: str,
    train_fnames: list[str],
    dv2: torch.nn.Module,
    upsampler: FeatureUpsampler,
    expr: Experiment,
) -> tuple[Classifier, object]:
    features, labels = [], []

    pca = None
    if train_cfg.add_dino_features:
        imgs = []
        for fname in sorted(listdir(f"{path}/{dataset}/images")):
            img_path = f"{path}/{dataset}/images/{fname}"
            arr = load_image(img_path)
            img = Image.fromarray(arr).convert("RGB")
            tr = closest_crop(img.height, img.width)
            tensor = convert_image(img, tr, device_str=DEVICE)
            imgs.append(tensor)

        _, pca = get_lr_feats(dv2, imgs, n_imgs=150, fit3d=True)

    for fname in train_fnames:
        img_path = f"{path}/{dataset}/images/{fname}.tif"
        labels_path = f"{path}/{dataset}/labels/{fname}.tif"

        img_arr = load_image(img_path)
        label_arr = load_labels(labels_path)

        feats = featurise_(img_arr, train_cfg.feature_config)
        if train_cfg.add_dino_features:
            img = Image.fromarray(img_arr).convert("RGB")
            deep_feats = get_deep_feats(img, dv2, upsampler, expr, 32, pca)
            feats = np.concatenate((feats, deep_feats), axis=-1)

        features.append(feats)
        labels.append(label_arr)

    print("Finished featurising")
    fit, target = get_training_data(features, labels)
    fit, target = shuffle_sample_training_data(fit, target, train_cfg.shuffle_data, train_cfg.n_samples)
    model = get_model(train_cfg.classifier, train_cfg.classifier_params, train_cfg.use_gpu)
    model = train(model, fit, target, None)
    return model, pca


def apply_model_over_images(
    dataset: AllowedDatasets,
    train_cfg: TrainingConfig,
    model: Classifier,
    path: str,
    dv2: torch.nn.Module,
    upsampler: FeatureUpsampler,
    expr: Experiment,
    verbose: bool = False,
    early_cutoff_n: int = -1,
    existing_pca: object | None = None,
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    img_fnames = sorted(listdir(f"{path}/{dataset}/images"))
    N_imgs = len(img_fnames)

    selected_imgs = img_fnames if early_cutoff_n <= 0 else img_fnames[:early_cutoff_n]

    for i, fname in enumerate(selected_imgs):
        if verbose and i % 10 == 0:
            print(f"[{i:02d}/{N_imgs}] - {fname}")
        img_path = f"{path}/{dataset}/images/{fname}"
        img_arr = load_image(img_path)

        feats = featurise_(img_arr, train_cfg.feature_config)
        if train_cfg.add_dino_features:
            img = Image.fromarray(img_arr).convert("RGB")
            deep_feats = get_deep_feats(img, dv2, upsampler, expr, 32, existing_pca)
            feats = np.concatenate((feats, deep_feats), axis=-1)

        pred, _ = apply(model, feats, train_cfg, image=img_arr)  # apply_(model, feats)
        preds[fname] = pred
    return preds


def eval_preds(dataset: AllowedDatasets, preds: dict[str, np.ndarray], path: str) -> float:
    mious: list[float] = []
    seg_fnames = sorted(listdir(f"{path}/{dataset}/segmentations"))
    for i, fname in enumerate(seg_fnames):
        if fname not in list(preds.keys()):
            continue
        seg_path = f"{path}/{dataset}/segmentations/{fname}"
        pred = preds[fname]
        ground_truth = load_labels(seg_path)
        miou = class_avg_miou(pred, ground_truth)
        mious.append(miou)
    return np.mean(mious)

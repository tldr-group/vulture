import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from os import listdir, path, makedirs
from shutil import copyfile
import time as time
import numpy as np

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
import pretrained_microscopy_models as pmm
import albumentations as albu

from interactive_seg_backend.utils import class_avg_miou, class_avg_mious

from typing import Any, Literal


def get_model(
    n_classes: int,
    arch: str = "UnetPlusPlus",
    encoder: str = "resnet50",
    pretrained_weights: str = "micronet",
    dict_path: str | None = None,
) -> nn.Module:
    # Create the Unet model with a resnet backbone that is pre-trained on micronet
    model = pmm.segmentation_training.create_segmentation_model(
        architecture=arch,
        encoder=encoder,
        encoder_weights=pretrained_weights,  # use encoder pre-trained on micronet
        classes=n_classes,  # secondary precipitates, tertiary precipitates, matrix
    )
    if dict_path is not None:
        state = torch.load(dict_path, weights_only=False)
        model.load_state_dict(pmm.util.remove_module_from_state_dict(state["state_dict"]))
    return model


def get_training_augmentation() -> albu.Compose:
    train_transform = [
        albu.VerticalFlip(p=0.75),
        albu.RandomRotate90(p=1),
        # albu.GaussNoise(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1, brightness_limit=0.25, contrast_limit=0.0),
                albu.RandomGamma(p=1),
            ],
            p=0.50,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                # albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.50,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1, brightness_limit=0, contrast_limit=0.3),
                albu.HueSaturationValue(p=1),
            ],
            p=0.50,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation() -> albu.Compose:
    """Add paddings to make image shape divisible by 32"""
    # This is turned off for this dataset
    test_transform = [
        # albu.Resize(height,width)
    ]
    return albu.Compose(test_transform)


def to_tensor(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn) -> albu.Compose:
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_dataset(
    class_values: dict[str, list[int]],
    image_paths_or_dir: str | list[str],
    mask_paths_or_dir: str | list[str],
    which: Literal["train", "val", "test"] = "train",
    encoder: str = "resnet50",
) -> pmm.io.Dataset:
    images_fps: list[str]
    masks_fps: list[str]
    if type(image_paths_or_dir) is str:
        assert type(mask_paths_or_dir) is str
        img_ids = listdir(image_paths_or_dir)
        images_fps = [path.join(image_paths_or_dir, image_id) for image_id in img_ids]
        mask_ids = listdir(mask_paths_or_dir)
        masks_fps = [path.join(mask_paths_or_dir, mask_id) for mask_id in mask_ids]
    else:
        assert type(mask_paths_or_dir) is list[str]
        assert type(image_paths_or_dir) is list[str]
        images_fps = image_paths_or_dir
        masks_fps = mask_paths_or_dir

    aug = get_training_augmentation() if which == "train" else get_validation_augmentation()
    preprocessing_fn = get_preprocessing(smp.encoders.get_preprocessing_fn(encoder, "imagenet"))
    dataset = pmm.io.Dataset(
        images=images_fps, masks=masks_fps, class_values=class_values, augmentation=aug, preprocessing=preprocessing_fn
    )
    return dataset


class MaskedDiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MaskedDiceBCELoss, self).__init__()
        self.weight = weight
        self.__name__ = "DiceBCELoss"

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        b, c, h, w = inputs.shape
        _, c_t, _, _ = targets.shape

        unlabelled = targets[:, -1, :, :].bool()
        labelled = ~unlabelled
        mask = labelled.view((b, 1, h * w))  # shape: (b, 1, h, w), keep dims for broadcasting
        if torch.sum(mask) == 0 or c_t == c:
            mask = torch.ones((b, 1, h * w), device=inputs.device, dtype=inputs.dtype).bool()
        # print(torch.sum(mask))

        # flatten label and prediction tensors
        inputs = inputs.view((b, c, h * w))
        targets = targets[:, :-1, :, :].view((b, c, h * w))

        valid_inputs = inputs[mask.expand_as(inputs)]
        valid_targets = targets[mask.expand_as(targets)]
        # print(valid_inputs.shape, valid_targets.shape)
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        intersection = (valid_inputs * valid_targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (valid_inputs.sum() + valid_targets.sum() + smooth)
        BCE = F.binary_cross_entropy(valid_inputs, valid_targets, reduction="mean")
        Dice_BCE = self.weight * BCE + (1 - self.weight) * dice_loss

        return Dice_BCE


def eval_miou(
    model: nn.Module, test_ds: pmm.io.Dataset, device: str = "cuda:0", is_sparse: bool = True, save_preds: bool = False
):
    mious = []
    preds, gts = [], []
    for n in range(len(test_ds)):
        image, gt_mask = test_ds[n]

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        n_ch, h, w = gt_mask.shape
        if is_sparse:
            n_ch -= 1

        gt_int_arr = np.zeros((h, w))
        pred_int_arr = np.zeros((h, w))

        ch_init = 0  # 1 if is_sparse else 0
        for i in range(ch_init, n_ch + ch_init):
            gt_int_arr = np.where(gt_mask[i] != 0, i, gt_int_arr)
            pred_int_arr = np.where(pr_mask[i] != 0, i, pred_int_arr)
        preds.append(pred_int_arr)
        gts.append(gt_int_arr)
        miou = class_avg_mious(pred_int_arr, gt_int_arr)
        mious.append(miou)
    miou = np.mean([np.mean(m) for m in mious])
    if save_preds:
        return {"miou": miou, "class_mious": mious, "gts": gts, "preds": preds}
    else:
        return {"miou": miou, "class_mious": mious}


def train_segmentation_model_with_eval(
    model: nn.Module | dict | str,
    architecture: str,
    encoder: str,
    train_dataset: pmm.io.Dataset,
    validation_dataset: pmm.io.Dataset,
    class_values: dict[str, list[int]],
    n_classes: int = 3,
    loss=None,
    epochs: int | None = None,
    save_per: int = 10,
    patience: int = 30,
    device: str = "cuda",
    lr: float = 2e-4,
    lr_decay: float = 0.00,
    batch_size: int = 20,
    val_batch_size: int = 12,
    num_workers: int = 0,
    save_folder="./",
    save_name: str | None = None,
    multi_gpu: bool = False,
):
    # setup and check parameters
    assert len(class_values) != 2, "Two classes is binary classification.  Just specify the posative class value."
    assert patience is not None or epochs is not None, "Need to set patience or epochs to define a stopping point."
    epochs = 1e10 if epochs is None else epochs  # this will basically never be reached.
    patience = 1e10 if patience is None else patience  # this will basically never be reached.

    # load or create model
    # TODO need to reload the optimizer.
    if type(model) is dict:  # passed the state back for restarting
        state = model  # load state dictionary
        architecture = state["architecture"]
        encoder = state["encoder"]
        # create empty model
        model = pmm.segmentation_training.create_segmentation_model(architecture, encoder, None, n_classes)
        # load saved weights
        model.load_state_dict(util.remove_module_from_state_dict(state["state_dict"]))
    elif type(model) is str:  # passed saved model state for restarting
        state = torch.load(model, weights_only=False)
        architecture = state["architecture"]
        encoder = state["encoder"]
        model = pmm.segmentation_training.create_segmentation_model(architecture, encoder, None, n_classes)
        model.load_state_dict(util.remove_module_from_state_dict(state["state_dict"]))
    else:  # Fresh PyTorch model for segmentation
        state = {
            "architecture": architecture,
            "encoder": encoder,
            "train_loss": [],
            "valid_loss": [],
            "train_iou": [],
            "valid_iou": [],
            "max_score": 0,
            "class_values": class_values,
        }

    if not path.exists(save_folder):
        makedirs(save_folder)

    if multi_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # create training dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        validation_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    loss = pmm.losses.DiceBCELoss(weight=0.7) if loss is None else loss

    # metrics = [
    #     smp_utils.metrics.IoU(threshold=0.5),
    # ]
    metrics = []

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
        ]
    )

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    patience_step = 0
    max_score = 0
    best_epoch = 0
    epoch = 0
    is_sparse = len(list(class_values.keys())) == 4
    tot_time = 0
    results: list[dict] = []

    t0 = time.time()

    while True:
        t = time.time() - t0
        print(
            "\nEpoch: {}, lr: {:0.8f}, time: {:0.2f} seconds, patience step: {}, best iou: {:0.4f}".format(
                epoch, lr, t, patience_step, max_score
            )
        )
        torch.cuda.synchronize("cuda")  # s.t time is accurate
        t0 = time.time()
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # update the state
        state["epoch"] = epoch + 1
        state["state_dict"] = model.state_dict()
        state["optimizer"] = optimizer.state_dict()
        state["train_loss"].append(train_logs["DiceBCELoss"])
        state["valid_loss"].append(valid_logs["DiceBCELoss"])
        torch.cuda.synchronize("cuda")  # s.t time is accurate
        t1 = time.time()

        tot_time += t1 - t0

        # save the model
        torch.save(state, path.join(save_folder, "checkpoint.pth.tar"))

        if epoch % save_per == 0:
            eval_result = eval_miou(model, validation_dataset, device, is_sparse, epoch >= epochs)
            eval_result["time"] = t1 - t0
            eval_result["tot_time"] = tot_time
            eval_result["epoch"] = epoch
            results.append(eval_result)

            print(f"[{epoch:3d}/{epochs}] ({tot_time:.3f}): {eval_result['miou']:.4f}")

        # Increment the epoch and decay the learning rate
        epoch += 1
        lr = optimizer.param_groups[0]["lr"] * (1 - lr_decay)
        optimizer.param_groups[0]["lr"] = lr

        if epoch >= epochs:
            print("\n\nTraining done! Saving final model")
            if save_name is not None:
                copyfile(path.join(save_folder, "model_best.pth.tar"), path.join(save_folder, save_name))
            return results, state

        # Use early stopping if there has not been improvment in a while
        if patience_step >= patience:
            print("\n\nTraining done!  No improvement in {} epochs. Saving final model".format(patience))
            if save_name is not None:
                copyfile(path.join(save_folder, "model_best.pth.tar"), path.join(save_folder, save_name))
            return results, state


# def iteratively_train_cnn(
#     train_ds: pmm.io.Dataset,
#     val_ds: pmm.io.Dataset,
#     loss,
#     class_values: dict[str, list[int]],
#     n_epochs: int,
#     save_per: int = 10,
#     save_fname: str = "UnetPlusPlus_resnet50_high_lr.pth.tar",
#     pretrained_weights: str = "micronet",
#     arch: str = "UnetPlusPlus",
#     encoder: str = "resnet50",
#     device: str = "cuda:0",
# ) -> tuple[dict[int, Any], Any]:
#     # TODO: this is stupid because of adam momemtum
#     model = pmm.segmentation_training.create_segmentation_model(
#         architecture=arch,
#         encoder=encoder,
#         encoder_weights=pretrained_weights,  # use encoder pre-trained on micronet
#         classes=3,  # secondary precipitates, tertiary precipitates, matrix
#     )
#     is_sparse = len(list(class_values.keys())) == 4

#     tot_time = 0
#     results: list[dict] = []
#     state = None
#     eval_result = None
#     for i in range(n_epochs):
#         model_state = model if i == 0 else state
#         #  TODO: we can mock the weight decay by manually decreasing LR in here
#         t0 = time.time()
#         state = pmm.segmentation_training.train_segmentation_model(
#             model=model_state,
#             loss=loss,
#             architecture=arch,
#             encoder=encoder,
#             train_dataset=train_ds,
#             validation_dataset=val_ds,
#             class_values=class_values,
#             patience=30,
#             device=device,
#             lr=2e-4,
#             batch_size=6,
#             val_batch_size=6,
#             save_folder="models",
#             epochs=1,
#             save_name=save_fname,
#         )
#         t1 = time.time()
#         tot_time += t1 - t0
#         # problem: only the state is being changed here, not the model

#         if i % save_per == 0:
#             model.load_state_dict(pmm.util.remove_module_from_state_dict(state["state_dict"]))
#             eval_result = eval_miou(model, val_ds, device, is_sparse)
#             eval_result["time"] = t1 - t0
#             eval_result["tot_time"] = tot_time
#             eval_result["epoch"] = i
#             results.append(eval_result)

#             print(f"[{i:3d}/{n_epochs}] ({tot_time:.3f}): {eval_result['miou']:.4f}")
#     return results, state

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from time import time_ns
from PIL import Image
import matplotlib.pyplot as plt


def measure_mem_time(
    img: torch.Tensor,
    feats: torch.Tensor,
    model: torch.nn.Module,
) -> tuple[float, float]:

    torch.cuda.reset_peak_memory_stats(feats.device)  # s.t memory is accurate
    torch.cuda.synchronize(feats.device)  # s.t time is accurate

    def _to_MB(x: int) -> float:
        return x / (1024**2)

    def _to_s(t: int) -> float:
        return t / 1e9

    start_m = torch.cuda.max_memory_allocated(feats.device)
    start_t = time_ns()

    model.forward(img, feats)

    end_m = torch.cuda.max_memory_allocated(feats.device)
    torch.cuda.synchronize(feats.device)
    end_t = time_ns()

    return _to_MB(end_m - start_m), _to_s(end_t - start_t)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if len(arr.shape) == 4:
        arr = arr[0]
    return arr


def get_arrs_from_batch(
    img: torch.Tensor,
    lr_feats: torch.Tensor,
    hr_feats: torch.Tensor,
    pred_hr_feats: torch.Tensor | None,
) -> list[list[np.ndarray]]:
    b, c, h, w = hr_feats.shape

    arrs: list[list[np.ndarray]] = []
    for i in range(b):
        img_tensor, lr_feat_tensor, hr_feat_tensor, pred_hr_tensor = (
            img[i],
            lr_feats[i],
            hr_feats[i],
            pred_hr_feats[i],
        )
        img_arr = to_numpy(img_tensor.permute((1, 2, 0)))

        out_2D_arrs: list[np.ndarray] = [img_arr]
        tensors = (
            (lr_feat_tensor, hr_feat_tensor, pred_hr_tensor)
            if isinstance(pred_hr_feats, torch.Tensor)
            else (lr_feat_tensor, hr_feat_tensor)
        )
        for i, d in enumerate(tensors):
            feat_arr = to_numpy(d)
            k = 3
            pca = PCA(n_components=k)

            n_c, h, w = feat_arr.shape
            data_flat = feat_arr.reshape((n_c, h * w)).T
            out = pca.fit_transform(data_flat)
            out_rescaled = MinMaxScaler().fit_transform(out)

            out_2D = out_rescaled.reshape((h, w, k))
            out_2D_arrs.append(out_2D)
        arrs.append(out_2D_arrs)
    return arrs


# put vis code in here
def visualise(
    img: torch.Tensor | Image.Image,
    lr_feats: torch.Tensor,
    hr_feats: torch.Tensor,
    pred_hr_feats: torch.Tensor | None,
    out_path: str,
) -> None:
    # b, c, h, w = hr_feats.shape
    n_rows = 4 if isinstance(pred_hr_feats, torch.Tensor) else 3
    arrs = get_arrs_from_batch(img, lr_feats, hr_feats, pred_hr_feats)
    fig, axs = plt.subplots(nrows=n_rows, ncols=len(arrs))
    fig.set_size_inches(24, 12)
    for i, arr in enumerate(arrs):
        for j, sub_arr in enumerate(arr):
            if len(arrs) == 1:
                axs[j].imshow(sub_arr)
                axs[j].set_axis_off()
            else:
                axs[j, i].imshow(sub_arr)
                axs[j, i].set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_losses(train_loss: list[float], val_loss: list[float], out_path: str) -> None:
    epochs = np.arange(len(train_loss))
    plt.semilogy(epochs, train_loss, lw=2, label="train")
    plt.semilogy(epochs, val_loss, lw=2, label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

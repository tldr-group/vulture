import torch
from time import time_ns


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

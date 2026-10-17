import random
import numpy as np
import torch


def select_device(requested: str = "auto") -> torch.device:
    """MPS is Apple's Metal backend for both training and inference."""
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable: use Apple Silicon, supported macOS and a Metal-enabled PyTorch build")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return torch.device(requested)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def device_report() -> dict:
    return {"torch": torch.__version__, "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
            "selected": str(select_device())}

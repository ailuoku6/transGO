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


def select_inference_device(requested="auto", backend="torch"):
    """Core ML owns its devices; PyTorch only loads its source weights on CPU."""
    if backend == "coreml":
        return torch.device("cpu")
    if backend != "torch":
        raise ValueError("inference backend must be torch or coreml")
    return select_device(requested)


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def device_report() -> dict:
    from .coreml import coreml_report
    return {"torch": torch.__version__, "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
            "selected": str(select_device()), "coreml": coreml_report()}

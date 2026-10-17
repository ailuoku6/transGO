import hashlib
import json
import platform
from pathlib import Path
import torch
from .config import from_dict
from .model import PolicyValueNet


TRAINING_ORIGIN = "random_init_selfplay"


def require_selfplay_origin(payload):
    if payload.get("training_origin") != TRAINING_ORIGIN:
        raise ValueError("checkpoint has no verified random-init selfplay origin; start a fresh Zero run")


def save_checkpoint(path, model, config, step=0, optimizer=None, rng=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "config": config.to_dict(), "step": step,
               "training_origin": TRAINING_ORIGIN,
               "software": {"torch": str(torch.__version__), "python": platform.python_version(),
                            "engine": "0.1.0", "feature_schema": 1},
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "torch_rng": torch.get_rng_state(),
               "optimizer": optimizer.state_dict() if optimizer else None,
               "numpy_rng": json.dumps(rng.bit_generator.state) if rng else None}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device, *, selfplay_only=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    if selfplay_only:
        require_selfplay_origin(payload)
    config = from_dict(payload["config"])
    model = PolicyValueNet(config.model)
    model.load_state_dict(payload["model"])
    return model.to(device), config, payload


def model_version(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

import pytest
import torch
from transformer_go._native import Board
from transformer_go.config import ModelConfig, load_config
from transformer_go.model import PolicyValueNet
from transformer_go.search import Evaluator
from transformer_go.runtime import select_device, seed_all
from transformer_go.selfplay import generate
from transformer_go.train import train
from transformer_go.checkpoint import load_checkpoint

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU unavailable in this environment")


@pytest.mark.parametrize("size", [9, 13, 19])
def test_metal_board_sizes_forward_backward(size):
    model = PolicyValueNet(ModelConfig(width=32, layers=2, heads=4)).to("mps")
    features = torch.from_numpy(Board(size).features()).unsqueeze(0).to("mps")
    out = model(features)
    loss = (out["policy_logits"].square().mean()+out["value_logits"].square().mean()+
            out["ownership_logits"].square().mean()+out["score_mean"].square().mean()+out["moves_left"].mean())
    loss.backward()
    torch.mps.synchronize()
    assert loss.device.type == "mps" and torch.isfinite(loss).item()
    assert all(p.grad is None or p.grad.device.type == "mps" for p in model.parameters())


def test_metal_selfplay_train_resume(tmp_path):
    config = load_config("configs/smoke.yaml")
    seed_all(config.seed)
    device = select_device("mps")
    model = PolicyValueNet(config.model).to(device)
    generate(config, Evaluator(model, device), tmp_path/"replay", 2, report=lambda _: None)
    path = tmp_path/"model.pt"
    train(config, model, device, tmp_path/"replay", path, 2, report=lambda _: None)
    restored, _, payload = load_checkpoint(path, device)
    result = train(config, restored, device, tmp_path/"replay", path, 1, payload, lambda _: None)
    assert result["step"] == 3 and result["device"] == "mps"

import torch
from transformer_go.config import ModelConfig
from transformer_go.model import PolicyValueNet


def test_model_shapes_and_mask():
    model = PolicyValueNet(ModelConfig(width=32, layers=2, heads=4, window=3))
    features = torch.zeros(2, 21, 3, 3)
    mask = torch.ones(2, 10, dtype=torch.bool); mask[:, 0] = False
    out = model(features, mask)
    assert out["policy_logits"].shape == (2, 10)
    assert torch.isneginf(out["policy_logits"][:, 0]).all()
    assert out["value_logits"].shape == (2, 3)
    assert out["ownership_logits"].shape == (2, 3, 3, 3)

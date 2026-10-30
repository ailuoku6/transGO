import pytest
import torch

from transformer_go.config import TrainConfig, load_config
from transformer_go.train import _ownership_targets, _smoothed_one_hot, _student_t_nll, losses


def test_project_training_config_uses_calibrated_loss_by_default():
    config = load_config("configs/tiny.yaml").train

    assert config.policy_soft_weight == 0.25
    assert config.policy_soft_exponent == 0.25
    assert config.value_label_smoothing == 0.02
    assert config.ownership_label_smoothing == 0.02
    assert config.ownership_neutral_smoothing == 0.15
    assert config.score_robust_weight == 0.25
    assert config.score_student_t_df == 5.0


def test_soft_policy_loss_uses_only_legal_actions_and_has_finite_gradients():
    config = TrainConfig(policy_soft_weight=0.5)
    legal = torch.tensor([[True, True, False]])
    logits = torch.tensor([[2.0, -1.0, float("-inf")]], requires_grad=True)
    batch = {
        "legal": legal,
        "policy": torch.tensor([[0.9, 0.1, 0.0]]),
        "value": torch.tensor([2]),
        "score": torch.zeros(1),
        "ownership": torch.ones(1, 2, 2, dtype=torch.long),
        "moves_left": torch.ones(1),
        "aux_mask": torch.ones(1),
    }
    outputs = {
        "policy_logits": logits,
        "value_logits": torch.zeros(1, 3),
        "score_mean": torch.zeros(1),
        "score_logvar": torch.zeros(1),
        "ownership_logits": torch.zeros(1, 2, 2, 3),
        "moves_left": torch.ones(1),
    }

    result = losses(outputs, batch, config)
    result["total"].backward()

    assert torch.isfinite(result["total"])
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0
    assert result["policy"] > 0


def test_student_t_score_nll_reduces_outlier_gradient_and_clamps_scale():
    residual_gaussian = torch.tensor([20.0], requires_grad=True)
    gaussian = 0.5 * residual_gaussian.square()
    gaussian.backward()

    residual_student = torch.tensor([20.0], requires_grad=True)
    robust = _student_t_nll(residual_student, torch.tensor([0.0]), 5.0)
    robust.backward()

    extreme_scale = _student_t_nll(torch.tensor([20.0]), torch.tensor([100.0]), 5.0)

    assert torch.isfinite(robust).all()
    assert torch.isfinite(extreme_scale).all()
    assert abs(residual_student.grad.item()) < abs(residual_gaussian.grad.item())


def test_label_smoothing_and_loss_blend_config_are_validated():
    assert TrainConfig(value_label_smoothing=0.0, score_robust_weight=0.0)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        TrainConfig(ownership_neutral_smoothing=1.1)


def test_wdl_and_neutral_ownership_targets_are_soft_and_normalized():
    value = _smoothed_one_hot(torch.tensor([2]), 3, 0.02)
    ownership = _ownership_targets(torch.tensor([[[1, 0]]]), 0.02, 0.15)

    torch.testing.assert_close(value.sum(-1), torch.ones(1))
    assert value[0, 2] < 1.0 and value[0, 0] > 0.0
    torch.testing.assert_close(ownership.sum(-1), torch.ones(1, 1, 2))
    assert ownership[0, 0, 0, 0] > 0.0 and ownership[0, 0, 0, 2] > 0.0
    assert ownership[0, 0, 1, 0] > ownership[0, 0, 1, 2]

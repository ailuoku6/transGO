import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_replay, augment
from .checkpoint import save_checkpoint, require_selfplay_origin
from .training_metrics import TrainingLog


def _normalize_policy(target, legal):
    """Project a policy target onto legal moves and renormalize it."""
    target = target.float().clamp_min(0.0) * legal
    return target / target.sum(-1, keepdim=True).clamp_min(torch.finfo(target.dtype).eps)


def _smoothed_one_hot(labels, classes, smoothing):
    target = F.one_hot(labels.long(), classes).to(dtype=torch.float32)
    return target * (1.0 - smoothing) + smoothing / classes


def _ownership_targets(labels, label_smoothing, neutral_smoothing):
    target = _smoothed_one_hot(labels, 3, label_smoothing)
    uncertain = torch.zeros_like(target)
    uncertain[..., 0] = 0.5
    uncertain[..., 2] = 0.5
    neutral_target = (1.0 - neutral_smoothing) * target + neutral_smoothing * uncertain
    return torch.where((labels == 1).unsqueeze(-1), neutral_target, target)


def _student_t_nll(residual, logvar, degrees_of_freedom):
    """A scale-aware heavy-tailed NLL for terminal score noise/outliers."""
    logvar = logvar.clamp(-8.0, 8.0)
    variance = torch.exp(logvar)
    normalization = (math.lgamma(degrees_of_freedom / 2.0) -
                     math.lgamma((degrees_of_freedom + 1.0) / 2.0) +
                     0.5 * math.log(degrees_of_freedom * math.pi))
    return (normalization + 0.5 * logvar +
            0.5 * (degrees_of_freedom + 1.0) * torch.log1p(
                residual.square() / (degrees_of_freedom * variance)))


def losses(outputs, batch, config):
    legal = batch["legal"].bool()
    target_policy = _normalize_policy(batch["policy"], legal)
    # 0 * -inf is NaN: explicitly zero masked log-probabilities before multiplication.
    logp = outputs["policy_logits"].log_softmax(-1).masked_fill(~legal, 0.0)
    policy_target_loss = -(target_policy * logp).sum(-1)
    # A fourth-root target is KataGo's low-visit policy auxiliary objective.
    soft_target = _normalize_policy(target_policy.clamp_min(1.0e-7).pow(config.policy_soft_exponent), legal)
    policy_soft_loss = -(soft_target * logp).sum(-1)
    policy = ((policy_target_loss + config.policy_soft_weight * policy_soft_loss) /
              (1.0 + config.policy_soft_weight)).mean()

    value_target = _smoothed_one_hot(batch["value"], 3, config.value_label_smoothing)
    value = -(value_target * F.log_softmax(outputs["value_logits"], dim=-1)).sum(-1).mean()
    mask = batch["aux_mask"]
    denominator = mask.sum().clamp_min(1)
    residual = outputs["score_mean"] - batch["score"]
    logvar = outputs["score_logvar"].clamp(-8.0, 8.0)
    gaussian = 0.5 * (residual.square() * torch.exp(-logvar) + logvar + math.log(2.0 * math.pi))
    robust = _student_t_nll(residual, logvar, config.score_student_t_df)
    score_samplewise = ((1.0 - config.score_robust_weight) * gaussian +
                        config.score_robust_weight * robust)
    score = (score_samplewise * mask).sum() / denominator

    ownership_target = _ownership_targets(
        batch["ownership"].long(),
        config.ownership_label_smoothing,
        config.ownership_neutral_smoothing,
    )
    ownership_logp = F.log_softmax(outputs["ownership_logits"], dim=-1)
    ownership_per_point = -(ownership_target * ownership_logp).sum(-1)
    ownership_per_sample = ownership_per_point.mean((1, 2))
    ownership = (ownership_per_sample * mask).sum() / denominator
    moves = (F.smooth_l1_loss(outputs["moves_left"], batch["moves_left"], reduction="none")*mask).sum()/denominator
    total = policy+value+config.score_weight*score+config.ownership_weight*ownership+config.moves_weight*moves
    return {"total": total, "policy": policy, "value": value, "score": score, "ownership": ownership, "moves": moves}


def train(config, model, device, directory, output, steps=100, resume=None, report=print,
          *, metrics_dir=None, iteration=None):
    if steps < 1 or config.train.batch_size < 1 or config.train.replay_positions < 1:
        raise ValueError("invalid training budget")
    if resume is not None:
        require_selfplay_origin(resume)
    replay, sampling_weights = load_replay(
        directory, config.board_size, config.train.replay_positions,
        return_sampling_weights=True)
    sampling_probability = sampling_weights / sampling_weights.sum()
    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    start = 0
    if resume is not None:
        if resume.get("optimizer") is not None:
            optimizer.load_state_dict(resume["optimizer"])
        start = resume["step"]
        if resume.get("numpy_rng"):
            rng.bit_generator.state = json.loads(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
    model.train()
    metrics = {}
    if metrics_dir is None:
        metrics_dir = Path(output).parent / "training" / Path(output).stem
    log = TrainingLog(metrics_dir, output, start, steps, iteration)
    try:
        for step in range(start+1, start+steps+1):
            indices = rng.choice(len(replay["value"]), size=config.train.batch_size,
                                 replace=True, p=sampling_probability)
            batch = augment({k: v[indices] for k, v in replay.items()}, rng)
            batch = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["features"].float(), batch["legal"])
            loss = losses(outputs, batch, config.train)
            if not torch.isfinite(loss["total"]):
                raise FloatingPointError("nonfinite training loss")
            loss["total"].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            metrics = {k: float(v.detach().cpu()) for k, v in loss.items()}
            metrics.update(step=step, grad_norm=float(grad.cpu()), positions=len(replay["value"]), device=str(device))
            log.record(metrics)
            if step == start+1 or step % 10 == 0 or step == start+steps:
                report(metrics)
        save_checkpoint(output, model, config, start+steps, optimizer, rng)
        metrics["summary"] = log.finish(metrics, {
            "policy": 1.0, "value": 1.0, "score": config.train.score_weight,
            "ownership": config.train.ownership_weight, "moves": config.train.moves_weight,
            "policy_soft_weight": config.train.policy_soft_weight,
            "policy_soft_exponent": config.train.policy_soft_exponent,
            "value_label_smoothing": config.train.value_label_smoothing,
            "ownership_label_smoothing": config.train.ownership_label_smoothing,
            "ownership_neutral_smoothing": config.train.ownership_neutral_smoothing,
            "score_robust_weight": config.train.score_robust_weight,
            "score_student_t_df": config.train.score_student_t_df})
        report(metrics["summary"])
    finally:
        log.close()
    return metrics

import json
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_replay, augment
from .checkpoint import save_checkpoint, require_selfplay_origin


def losses(outputs, batch, config):
    # 0 * -inf is NaN: explicitly zero masked log-probabilities before multiplication.
    logp = outputs["policy_logits"].log_softmax(-1).masked_fill(~batch["legal"].bool(), 0.0)
    policy = -(batch["policy"] * logp).sum(-1).mean()
    value = F.cross_entropy(outputs["value_logits"], batch["value"].long())
    mask = batch["aux_mask"]
    denominator = mask.sum().clamp_min(1)
    residual = outputs["score_mean"]-batch["score"]
    variance = outputs["score_logvar"]
    score = ((0.5*(residual.square()*torch.exp(-variance)+variance))*mask).sum()/denominator
    ownership = F.cross_entropy(outputs["ownership_logits"].permute(0, 3, 1, 2),
                                batch["ownership"].long(), reduction="none").mean((1, 2))
    ownership = (ownership*mask).sum()/denominator
    moves = (F.smooth_l1_loss(outputs["moves_left"], batch["moves_left"], reduction="none")*mask).sum()/denominator
    total = policy+value+config.score_weight*score+config.ownership_weight*ownership+config.moves_weight*moves
    return {"total": total, "policy": policy, "value": value, "score": score, "ownership": ownership, "moves": moves}


def train(config, model, device, directory, output, steps=100, resume=None, report=print):
    if steps < 1 or config.train.batch_size < 1 or config.train.replay_positions < 1:
        raise ValueError("invalid training budget")
    if resume is not None:
        require_selfplay_origin(resume)
    replay = load_replay(directory, config.board_size, config.train.replay_positions)
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
    for step in range(start+1, start+steps+1):
        indices = rng.integers(len(replay["value"]), size=config.train.batch_size)
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
        if step == start+1 or step % 10 == 0 or step == start+steps:
            report(metrics)
    save_checkpoint(output, model, config, start+steps, optimizer, rng)
    return metrics

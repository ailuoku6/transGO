import json
from pathlib import Path
import shutil
import torch
from .model import PolicyValueNet
from .search import Evaluator
from .checkpoint import save_checkpoint, load_checkpoint, model_version
from .selfplay import generate, validate_min_moves
from .train import train
from .evaluate import evaluate, evaluate_parallel
from .gating import (DEFAULT_GATING_GAMES, DEFAULT_REQUIRED_WIN_PROP, GATEKEEPER_PROTOCOL,
                     gatekeeper_decision, validate_gating_options)


def atomic_copy(source, target):
    target = Path(target)
    temporary = target.with_suffix(target.suffix+".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def should_promote(result, threshold=DEFAULT_REQUIRED_WIN_PROP):
    """Use match points, including only mathematically conclusive early stops."""
    tallied = result["wins"] + result["losses"] + result["draws"] + result.get("no_results", 0)
    if tallied != result["completed"] or tallied != result.get("played", tallied):
        return False
    # Historical unfinished games did not contribute points. New move-limit
    # games are explicitly scored using BoardHistory, just like KataGo.
    if result.get("truncated", 0) != result.get("forced_scored", 0):
        return False
    points = result["wins"] + 0.5 * (result["draws"] + result.get("no_results", 0))
    return gatekeeper_decision(points, tallied, result["games"], threshold) == "accepted"


def run_pipeline(config, device, directory, iterations=1, selfplay_games=8, train_steps=100,
                 eval_games=DEFAULT_GATING_GAMES, promote_threshold=DEFAULT_REQUIRED_WIN_PROP, report=print, min_moves=0,
                 score_utility=0.0, selfplay_workers=1, eval_workers=1):
    if iterations < 1:
        raise ValueError("invalid iteration count")
    validate_gating_options(eval_games, promote_threshold)
    validate_min_moves(config.board_size, min_moves)
    if not isinstance(selfplay_workers, int) or isinstance(selfplay_workers, bool) or selfplay_workers < 1:
        raise ValueError("selfplay_workers must be a positive integer")
    if not isinstance(eval_workers, int) or isinstance(eval_workers, bool) or eval_workers < 1:
        raise ValueError("eval_workers must be a positive integer")
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    best_path = root/"best.pt"
    learner_path = root/"learner.pt"
    if not best_path.exists():
        initial_device = torch.device("cpu") if selfplay_workers > 1 or eval_workers > 1 else device
        save_checkpoint(best_path, PolicyValueNet(config.model).to(initial_device), config)
    if not learner_path.exists(): atomic_copy(best_path, learner_path)
    for _ in range(iterations):
        # Parallel actors load their own copy, so keep the parent free of an
        # extra MPS model during self-play.  Serial mode reuses this model.
        load_device = torch.device("cpu") if selfplay_workers > 1 else device
        model, best_config, _ = load_checkpoint(best_path, load_device, selfplay_only=True)
        if best_config.to_dict() != config.to_dict(): raise ValueError("pipeline checkpoint config mismatch")
        index = max((int(p.stem.split("-")[1]) for p in root.glob("iteration-*.json")), default=0)+1
        if selfplay_workers > 1:
            from .actors import generate_parallel
            generate_parallel(config, best_path, root/"replay", selfplay_games,
                              selfplay_workers, config.seed+index*100000,
                              min_moves=min_moves, score_utility=score_utility,
                              report=report, device=device)
            del model
            model = None
        else:
            generate(config, Evaluator(model, device), root/"replay", selfplay_games,
                     config.seed+index*100000, model_version(best_path), report,
                     min_moves=min_moves, score_utility=score_utility)
        candidate = root/f"candidate-{index:04d}.pt"
        # Learner keeps improving even when a candidate fails to replace the actor model.
        model, learner_config, payload = load_checkpoint(learner_path, device, selfplay_only=True)
        if learner_config.to_dict() != config.to_dict(): raise ValueError("learner configuration mismatch")
        train(config, model, device, root/"replay", candidate, train_steps, payload, report)
        atomic_copy(candidate, learner_path)
        baseline_sha256 = model_version(best_path)
        # A reproducible new match distribution for each iteration, including
        # when resuming an existing run. Keep Config/checkpoint schema unchanged.
        eval_seed = config.seed + index * 100000 + 50000
        if eval_workers > 1:
            # The candidate checkpoint is the source of truth for workers;
            # release the parent copy before several workers load it on MPS.
            del model
            result = evaluate_parallel(
                config, candidate, best_path, eval_games, eval_workers, report,
                score_utility=score_utility,
                required_candidate_win_prop=promote_threshold, seed=eval_seed,
                device=device)
        else:
            baseline, _, _ = load_checkpoint(best_path, device)
            result = evaluate(config, Evaluator(model, device), Evaluator(baseline, device),
                              eval_games, report, score_utility=score_utility,
                              required_candidate_win_prop=promote_threshold, seed=eval_seed)
        promoted = should_promote(result, promote_threshold)
        if promoted: atomic_copy(candidate, best_path)
        record = {"iteration": index, "candidate": str(candidate), "promoted": promoted,
                  "best_sha256": model_version(best_path), "evaluation": result, "min_moves": min_moves,
                  "score_utility": score_utility, "promotion_policy": GATEKEEPER_PROTOCOL,
                  "required_candidate_win_prop": promote_threshold, "eval_seed": eval_seed,
                  "baseline_sha256": baseline_sha256, "candidate_sha256": model_version(candidate),
                  "selfplay_workers": selfplay_workers, "eval_workers": eval_workers}
        (root/f"iteration-{index:04d}.json").write_text(json.dumps(record, indent=2))
        report(record)

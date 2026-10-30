import json
from pathlib import Path
import shutil
import torch
from .model import PolicyValueNet
from .runtime import select_inference_device
from .search import Evaluator
from .checkpoint import save_checkpoint, load_checkpoint, model_version
from .selfplay import generate, validate_min_moves, MAX_ATTEMPTS_PER_GAME
from .train import train
from .training_metrics import append_loss_history
from .exploration import validate_profile, exploration_settings
from .evaluate import evaluate, evaluate_parallel
from .inference import validate_batching
from .gating import (DEFAULT_GATING_GAMES, DEFAULT_REQUIRED_WIN_PROP, GATEKEEPER_PROTOCOL,
                     gatekeeper_decision, validate_gating_options)

SELFPLAY_SEED_ITERATION_STRIDE = 1_000_000
SELFPLAY_SEED_OFFSET = 1_000  # Separate resumed runs from the old overlapping seed schedule.


def selfplay_iteration_seed(config_seed, iteration):
    return config_seed + iteration * SELFPLAY_SEED_ITERATION_STRIDE + SELFPLAY_SEED_OFFSET


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
                 score_utility=0.0, selfplay_workers=1, eval_workers=1,
                 inference_batch_size=256, inference_wait_ms=2.0, selfplay_policy="katago",
                 evaluator_options=None):
    if iterations < 1:
        raise ValueError("invalid iteration count")
    validate_gating_options(eval_games, promote_threshold)
    validate_min_moves(config.board_size, min_moves)
    validate_batching(inference_batch_size, inference_wait_ms)
    validate_profile(selfplay_policy)
    if selfplay_games * MAX_ATTEMPTS_PER_GAME > SELFPLAY_SEED_ITERATION_STRIDE - SELFPLAY_SEED_OFFSET:
        raise ValueError("too many self-play games for non-overlapping iteration seeds")
    options = evaluator_options or {}
    inference_device = select_inference_device(str(device), options.get("backend", "torch"))
    if not isinstance(selfplay_workers, int) or isinstance(selfplay_workers, bool) or selfplay_workers < 1:
        raise ValueError("selfplay_workers must be a positive integer")
    if not isinstance(eval_workers, int) or isinstance(eval_workers, bool) or eval_workers < 1:
        raise ValueError("eval_workers must be a positive integer")
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    best_path = root/"best.pt"
    learner_path = root/"learner.pt"
    if not best_path.exists():
        initial_device = torch.device("cpu") if selfplay_workers > 1 or eval_workers > 1 else inference_device
        save_checkpoint(best_path, PolicyValueNet(config.model).to(initial_device), config)
    if not learner_path.exists(): atomic_copy(best_path, learner_path)
    for _ in range(iterations):
        # Validate the checkpoint on CPU before central inference loads it
        # onto the requested device. Serial mode reuses this model.
        load_device = torch.device("cpu") if selfplay_workers > 1 else inference_device
        model, best_config, _ = load_checkpoint(best_path, load_device, selfplay_only=True)
        if best_config.to_dict() != config.to_dict(): raise ValueError("pipeline checkpoint config mismatch")
        index = max((int(p.stem.split("-")[1]) for p in root.glob("iteration-*.json")), default=0)+1
        generation_seed = selfplay_iteration_seed(config.seed, index)
        if selfplay_workers > 1:
            from .actors import generate_parallel
            del model
            generate_parallel(config, best_path, root/"replay", selfplay_games,
                              selfplay_workers, generation_seed,
                              min_moves=min_moves, score_utility=score_utility,
                              report=report, device=device,
                              inference_batch_size=inference_batch_size,
                              inference_wait_ms=inference_wait_ms, selfplay_policy=selfplay_policy,
                              evaluator_options=evaluator_options)
        else:
            generate(config, Evaluator(model, inference_device, **options), root/"replay", selfplay_games,
                     generation_seed, model_version(best_path), report,
                     min_moves=min_moves, score_utility=score_utility, selfplay_policy=selfplay_policy)
        candidate = root/f"candidate-{index:04d}.pt"
        # Learner keeps improving even when a candidate fails to replace the actor model.
        model, learner_config, payload = load_checkpoint(learner_path, device, selfplay_only=True)
        if learner_config.to_dict() != config.to_dict(): raise ValueError("learner configuration mismatch")
        training = train(config, model, device, root/"replay", candidate, train_steps, payload, report,
                         metrics_dir=root / "training" / f"iteration-{index:04d}", iteration=index)
        append_loss_history(root / "training-loss.csv", training["summary"])
        atomic_copy(candidate, learner_path)
        baseline_sha256 = model_version(best_path)
        # A reproducible new match distribution for each iteration, including
        # when resuming an existing run. Keep Config/checkpoint schema unchanged.
        eval_seed = config.seed + index * 100000 + 50000
        eval_recording = {"sgf_dir": root / "evaluation" / f"iteration-{index:04d}",
                          "candidate_name": f"{candidate.name} sha256={model_version(candidate)}",
                          "baseline_name": f"best.pt sha256={baseline_sha256}"}
        if eval_workers > 1:
            # Release the training model before the central service loads
            # one candidate and one baseline for all CPU search workers.
            del model
            result = evaluate_parallel(
                config, candidate, best_path, eval_games, eval_workers, report,
                score_utility=score_utility,
                required_candidate_win_prop=promote_threshold, seed=eval_seed,
                device=device, inference_batch_size=inference_batch_size,
                inference_wait_ms=inference_wait_ms, evaluator_options=evaluator_options, **eval_recording)
        else:
            baseline, _, _ = load_checkpoint(best_path, inference_device)
            result = evaluate(config, Evaluator(model, inference_device, **options),
                              Evaluator(baseline, inference_device, **options),
                              eval_games, report, score_utility=score_utility,
                              required_candidate_win_prop=promote_threshold, seed=eval_seed,
                              **eval_recording)
        promoted = should_promote(result, promote_threshold)
        if promoted: atomic_copy(candidate, best_path)
        record = {"iteration": index, "candidate": str(candidate), "promoted": promoted,
                  "training": training["summary"],
                  "selfplay_policy": selfplay_policy, "exploration": exploration_settings(config, selfplay_policy),
                  "best_sha256": model_version(best_path), "evaluation": result, "min_moves": min_moves,
                  "score_utility": score_utility, "promotion_policy": GATEKEEPER_PROTOCOL,
                  "required_candidate_win_prop": promote_threshold, "eval_seed": eval_seed,
                  "baseline_sha256": baseline_sha256, "candidate_sha256": model_version(candidate),
                  "selfplay_workers": selfplay_workers, "eval_workers": eval_workers,
                  "inference_batch_size": inference_batch_size,
                  "inference_backend": options.get("backend", "torch"),
                  "coreml_batch_size": options.get("coreml_batch_size", 16) if options.get("backend") == "coreml" else None,
                  "inference_wait_ms": inference_wait_ms, "tree_reuse": True}
        (root/f"iteration-{index:04d}.json").write_text(json.dumps(record, indent=2))
        report(record)

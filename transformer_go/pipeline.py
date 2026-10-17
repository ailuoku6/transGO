import json
from pathlib import Path
import shutil
from .model import PolicyValueNet
from .search import Evaluator
from .checkpoint import save_checkpoint, load_checkpoint, model_version
from .selfplay import generate, validate_min_moves
from .train import train
from .evaluate import evaluate


def atomic_copy(source, target):
    target = Path(target)
    temporary = target.with_suffix(target.suffix+".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def should_promote(result, threshold=0.55, min_games=40):
    return (result["truncated"] == 0 and result["completed"] >= min_games
            and result["winrate"] >= threshold and result["confidence95"][0] > 0.5)


def run_pipeline(config, device, directory, iterations=1, selfplay_games=8, train_steps=100,
                 eval_games=40, promote_threshold=0.55, report=print, min_moves=0,
                 score_utility=0.0):
    if iterations < 1 or not 0.5 <= promote_threshold <= 1:
        raise ValueError("invalid iteration count/promotion threshold")
    validate_min_moves(config.board_size, min_moves)
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    best_path = root/"best.pt"
    learner_path = root/"learner.pt"
    if not best_path.exists(): save_checkpoint(best_path, PolicyValueNet(config.model).to(device), config)
    if not learner_path.exists(): atomic_copy(best_path, learner_path)
    for _ in range(iterations):
        model, best_config, _ = load_checkpoint(best_path, device, selfplay_only=True)
        if best_config.to_dict() != config.to_dict(): raise ValueError("pipeline checkpoint config mismatch")
        index = max((int(p.stem.split("-")[1]) for p in root.glob("iteration-*.json")), default=0)+1
        generate(config, Evaluator(model, device), root/"replay", selfplay_games,
                 config.seed+index*100000, model_version(best_path), report,
                 min_moves=min_moves, score_utility=score_utility)
        candidate = root/f"candidate-{index:04d}.pt"
        # Learner keeps improving even when a candidate fails to replace the actor model.
        model, learner_config, payload = load_checkpoint(learner_path, device, selfplay_only=True)
        if learner_config.to_dict() != config.to_dict(): raise ValueError("learner configuration mismatch")
        train(config, model, device, root/"replay", candidate, train_steps, payload, report)
        atomic_copy(candidate, learner_path)
        baseline, _, _ = load_checkpoint(best_path, device)
        result = evaluate(config, Evaluator(model, device), Evaluator(baseline, device), eval_games, report,
                          score_utility=score_utility)
        # Reject incomplete evaluations and require both threshold and statistical evidence.
        promoted = should_promote(result, promote_threshold)
        if promoted: atomic_copy(candidate, best_path)
        record = {"iteration": index, "candidate": str(candidate), "promoted": promoted,
                  "best_sha256": model_version(best_path), "evaluation": result, "min_moves": min_moves,
                  "score_utility": score_utility}
        (root/f"iteration-{index:04d}.json").write_text(json.dumps(record, indent=2))
        report(record)

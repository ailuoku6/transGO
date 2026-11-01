import json
import math
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
from .regression_guard import apply_regression_guard

SELFPLAY_SEED_ITERATION_STRIDE = 1_000_000
SELFPLAY_SEED_OFFSET = 1_000  # Separate resumed runs from the old overlapping seed schedule.
ALWAYS_PROMOTE_PROTOCOL = "always-promote-v1"
ANCHOR_PROTOCOL = "fixed-anchor-v1"
DEFAULT_ANCHOR_GAMES = 100


def selfplay_iteration_seed(config_seed, iteration):
    return config_seed + iteration * SELFPLAY_SEED_ITERATION_STRIDE + SELFPLAY_SEED_OFFSET


def anchor_match_seed(config_seed, iteration):
    return config_seed + iteration * 100000 + 70000


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


def validate_anchor_options(config, anchor_checkpoint, anchor_every, anchor_games, rollback_drop=None):
    if not isinstance(anchor_every, int) or isinstance(anchor_every, bool) or anchor_every < 0:
        raise ValueError("anchor_every must be a nonnegative integer")
    if rollback_drop is not None:
        if (isinstance(rollback_drop, bool) or not isinstance(rollback_drop, (int, float))
                or not math.isfinite(rollback_drop) or not 0 < rollback_drop <= 1):
            raise ValueError("rollback_drop must be in (0, 1]")
        if not anchor_every:
            raise ValueError("rollback_drop requires anchor_every")
    if not anchor_every:
        return
    validate_gating_options(anchor_games)
    if anchor_checkpoint is None:
        raise ValueError("anchor_every requires anchor_checkpoint")
    if not Path(anchor_checkpoint).is_file():
        raise ValueError(f"anchor checkpoint does not exist: {anchor_checkpoint}")
    _, anchor_config, _ = load_checkpoint(anchor_checkpoint, torch.device("cpu"), selfplay_only=True)
    if (anchor_config.model != config.model or anchor_config.board_size != config.board_size):
        raise ValueError("anchor checkpoint architecture or board size differs from config")


def snapshot_anchor(root, anchor_checkpoint):
    """Copy the anchor into the run so edits to the source cannot move the yardstick."""
    digest = model_version(anchor_checkpoint)
    target = Path(root) / "anchors" / f"anchor-{digest[:16]}.pt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(anchor_checkpoint, target)
    if model_version(target) != digest:
        raise ValueError(f"anchor snapshot {target} does not match {anchor_checkpoint}")
    return target, digest


def _anchor_match(config, subject_path, anchor_path, games, workers, report, *, seed, score_utility,
                  device, inference_device, options, evaluator_options, inference_batch_size,
                  inference_wait_ms, recording):
    """Full fixed-length match without early stopping; it never affects promotion."""
    if workers > 1:
        return evaluate_parallel(
            config, subject_path, anchor_path, games, workers, report,
            score_utility=score_utility, seed=seed, device=device,
            inference_batch_size=inference_batch_size, inference_wait_ms=inference_wait_ms,
            evaluator_options=evaluator_options, **recording)
    subject, _, _ = load_checkpoint(subject_path, inference_device, selfplay_only=True)
    anchor, _, _ = load_checkpoint(anchor_path, inference_device, selfplay_only=True)
    return evaluate(config, Evaluator(subject, inference_device, **options),
                    Evaluator(anchor, inference_device, **options), games, report,
                    score_utility=score_utility, seed=seed, **recording)


def run_pipeline(config, device, directory, iterations=1, selfplay_games=8, train_steps=100,
                 eval_games=DEFAULT_GATING_GAMES, promote_threshold=DEFAULT_REQUIRED_WIN_PROP, report=print, min_moves=0,
                 score_utility=0.0, selfplay_workers=1, eval_workers=1,
                 inference_batch_size=256, inference_wait_ms=2.0, selfplay_policy="katago",
                 evaluator_options=None, *, gating=True, anchor_checkpoint=None, anchor_every=0,
                 anchor_games=DEFAULT_ANCHOR_GAMES, rollback_drop=None):
    """Run self-play, training and promotion rounds.

    gating=False promotes every candidate without a match, like AlphaZero.
    anchor_every=N plays the current best against a fixed anchor snapshot every
    N iterations. With rollback_drop set, a win-rate drop of at least that much
    below the best anchor result restores best.pt and learner.pt from the
    restore point saved at that result (see regression_guard).
    """
    if iterations < 1:
        raise ValueError("invalid iteration count")
    validate_gating_options(eval_games, promote_threshold)
    validate_anchor_options(config, anchor_checkpoint, anchor_every, anchor_games, rollback_drop)
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
    anchor_path, anchor_sha256 = (snapshot_anchor(root, anchor_checkpoint) if anchor_every
                                  else (None, None))
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
        eval_seed = result = None
        if not gating:
            del model
            promoted = True
        else:
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
                    inference_wait_ms=inference_wait_ms, evaluator_options=evaluator_options,
                    **eval_recording)
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
                  "score_utility": score_utility,
                  "promotion_policy": GATEKEEPER_PROTOCOL if gating else ALWAYS_PROMOTE_PROTOCOL,
                  "required_candidate_win_prop": promote_threshold if gating else None,
                  "eval_seed": eval_seed,
                  "baseline_sha256": baseline_sha256, "candidate_sha256": model_version(candidate),
                  "selfplay_workers": selfplay_workers, "eval_workers": eval_workers,
                  "inference_batch_size": inference_batch_size,
                  "inference_backend": options.get("backend", "torch"),
                  "coreml_batch_size": options.get("coreml_batch_size", 16) if options.get("backend") == "coreml" else None,
                  "inference_wait_ms": inference_wait_ms, "tree_reuse": True}
        record_path = root/f"iteration-{index:04d}.json"
        # Write before the optional anchor match so an interrupted check never
        # repeats self-play/training for an iteration that already finished.
        record_path.write_text(json.dumps(record, indent=2))
        if anchor_every and index % anchor_every == 0:
            record["anchor_evaluation"] = _run_anchor_check(
                config, root, index, best_path, anchor_path, anchor_sha256, anchor_games,
                eval_workers, report, score_utility=score_utility, device=device,
                inference_device=inference_device, options=options,
                evaluator_options=evaluator_options, inference_batch_size=inference_batch_size,
                inference_wait_ms=inference_wait_ms)
            if rollback_drop is not None:
                record["regression_guard"] = apply_regression_guard(
                    root, index, best_path, learner_path, anchor_sha256,
                    record["anchor_evaluation"]["result"]["winrate"], rollback_drop, report)
                record["rolled_back"] = record["regression_guard"]["decision"] == "rollback"
                if record["rolled_back"]:
                    # best_sha256 always names the weights the next self-play will use.
                    record["best_sha256_before_rollback"] = record["best_sha256"]
                    record["best_sha256"] = model_version(best_path)
                _append_guard_history(root, record["regression_guard"])
            record_path.write_text(json.dumps(record, indent=2))
        report(record)


def _append_guard_history(root, guard):
    entry = {"iteration": guard["iteration"], "decision": guard["decision"],
             "winrate": guard["winrate"], "reference_iteration": guard["restore_point"]["iteration"],
             "reference_winrate": guard["restore_point"]["winrate"]}
    if guard["decision"] == "rollback":
        entry.update(restored_best_sha256=guard["after"]["best_sha256"],
                     restored_learner_sha256=guard["after"]["learner_sha256"])
    with (root / "regression-guard.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")


def _run_anchor_check(config, root, index, best_path, anchor_path, anchor_sha256, games, workers,
                      report, **kwargs):
    subject_sha256 = model_version(best_path)
    seed = anchor_match_seed(config.seed, index)
    report({"event": "anchor_evaluation_start", "iteration": index, "games": games,
            "subject_sha256": subject_sha256, "anchor_sha256": anchor_sha256, "seed": seed})
    recording = {"sgf_dir": root / "evaluation" / f"anchor-{index:04d}",
                 "candidate_name": f"best.pt sha256={subject_sha256}",
                 "baseline_name": f"{anchor_path.name} sha256={anchor_sha256}"}
    result = _anchor_match(config, best_path, anchor_path, games, workers, report,
                           seed=seed, recording=recording, **kwargs)
    summary = {"protocol": ANCHOR_PROTOCOL, "iteration": index, "subject": "best.pt",
               "subject_sha256": subject_sha256, "anchor": str(anchor_path),
               "anchor_sha256": anchor_sha256, "seed": seed, "result": result}
    compact = {"iteration": index, "subject_sha256": subject_sha256, "anchor_sha256": anchor_sha256,
               **{key: result[key] for key in ("games", "played", "wins", "losses", "draws",
                                               "no_results", "forced_scored", "winrate",
                                               "confidence95", "elo_difference")}}
    with (root / "anchor-evaluations.jsonl").open("a") as stream:
        stream.write(json.dumps(compact) + "\n")
    report({"event": "anchor_evaluation", **compact})
    return summary

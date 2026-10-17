import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from ._native import Board
from .config import from_dict
from .checkpoint import load_checkpoint
from .runtime import select_device
from .search import search, choose_action
from .search import Evaluator
from .gating import (DEFAULT_GATING_GAMES, GATEKEEPER_PROTOCOL,
                     gatekeeper_decision, validate_gating_options)


_EVAL_STATE = None


def wilson_interval(wins, games, z=1.96):
    p = wins/games
    center = (p+z*z/(2*games))/(1+z*z/games)
    radius = z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/(1+z*z/games)
    return max(0, center-radius), min(1, center+radius)


def gatekeeper_temperature(move_number, board_size):
    """KataGo gatekeeper1.cfg: early=0.5, late=0.2, half-life=19.

    Search::interpolateEarly scales the half-life by board width / 19.
    This reproduces that schedule on our square boards.
    """
    return 0.2 + 0.3 * 0.5 ** (move_number / board_size)


def _play_evaluation_game(config, candidate, baseline, candidate_color, seed,
                          score_utility, max_moves):
    board = Board(config.board_size, config.komi, config.rules)
    rng = np.random.default_rng(seed)
    for _ in range(max_moves):
        evaluator = candidate if board.to_play == candidate_color else baseline
        result = search(board, evaluator, config.search, seed=int(rng.integers(2**31)),
                        noise=False, min_moves=0, score_utility=score_utility)
        action = choose_action(result["visits"],
                               gatekeeper_temperature(board.move_number, board.size), rng)
        board = board.play(action)
        if board.terminal():
            break
    forced_scored = not board.terminal()
    no_result = board.no_result()
    # Board.result() scores a copy using KataGo BoardHistory::endAndScoreGameNow,
    # matching gatekeeper's handling of a game that reaches maxMovesPerGame.
    winner = 0.0 if no_result else board.result() * candidate_color
    outcome = "no_result" if no_result else ("win" if winner > 0 else ("loss" if winner < 0 else "draw"))
    return {"outcome": outcome, "forced_scored": forced_scored, "moves": board.move_number}


def _init_evaluation_worker(raw, candidate_path, baseline_path, requested_device=None):
    """Load both networks once in each spawned evaluation worker."""
    global _EVAL_STATE
    config = from_dict(raw)
    requested = config.device if requested_device is None else str(requested_device)
    device = select_device(requested)
    candidate, candidate_config, _ = load_checkpoint(candidate_path, device, selfplay_only=True)
    baseline, baseline_config, _ = load_checkpoint(baseline_path, device, selfplay_only=True)
    candidate_raw = candidate_config.to_dict()
    baseline_raw = baseline_config.to_dict()
    requested_raw = config.to_dict()
    # Device, optimizer and training hyperparameters are runtime choices for
    # evaluation.  The two networks must share their architecture, and the
    # requested board size must match their dynamic output geometry.
    if (candidate_raw["model"] != baseline_raw["model"] or
            candidate_raw["board_size"] != baseline_raw["board_size"]):
        raise ValueError("candidate and baseline checkpoint configurations differ")
    if (candidate_raw["model"] != requested_raw["model"] or
            candidate_raw["board_size"] != requested_raw["board_size"]):
        raise ValueError("evaluation checkpoint configuration differs from requested config")
    _EVAL_STATE = (config, Evaluator(candidate, device), Evaluator(baseline, device))


def _evaluation_game_task(candidate_color, seed, score_utility, max_moves):
    if _EVAL_STATE is None:
        raise RuntimeError("evaluation worker was not initialized")
    config, candidate, baseline = _EVAL_STATE
    return _play_evaluation_game(config, candidate, baseline, candidate_color, seed,
                                 score_utility, max_moves)


def _validate_evaluation_args(config, games, score_utility, required_candidate_win_prop,
                              seed, max_moves):
    validate_gating_options(games, required_candidate_win_prop)
    if not math.isfinite(score_utility) or score_utility < 0:
        raise ValueError("score_utility must be finite and nonnegative")
    if not isinstance(max_moves, int) or isinstance(max_moves, bool) or max_moves < 1:
        raise ValueError("max_moves must be a positive integer")
    seed = config.seed if seed is None else seed
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("evaluation seed must be a nonnegative integer")
    return seed


def _evaluation_schedule(games, seed):
    """Build the deterministic color/seed schedule before starting workers."""
    rng = np.random.default_rng(seed)
    schedule = []
    colors = None
    for g in range(games):
        if g % 2 == 0:
            colors = rng.permutation([1, -1]).tolist()
        schedule.append((colors[g % 2], int(rng.integers(2**31))))
    return schedule


def _play_settings(config, max_moves):
    return {"board_size": config.board_size, "komi": config.komi, "rules": config.rules,
            "max_moves": max_moves, "opening": "empty", "temperature_early": 0.5,
            "temperature_late": 0.2, "temperature_halflife_19x19": 19,
            "root_noise": False}


def _finish_evaluation(config, games, seed, required_candidate_win_prop, score_utility,
                       max_moves, report, play_game, parallel_workers=1):
    """Run and aggregate a match; ``play_game`` is called with schedule entries."""
    gating = required_candidate_win_prop is not None
    target = games * required_candidate_win_prop if gating else None
    settings = _play_settings(config, max_moves)
    if parallel_workers > 1:
        settings["workers"] = parallel_workers
        settings["scheduler"] = "spawn-process-batches"
    report({"event": "evaluation_start", "games": games, "seed": seed,
            "protocol": GATEKEEPER_PROTOCOL if gating else "full-match-v2",
            "required_candidate_win_prop": required_candidate_win_prop,
            "target_points": target, "score_utility": score_utility,
            "play_settings": settings})
    wins = losses = draws = no_results = forced_scored = 0
    decision = "pending" if gating else None
    stop_reason = "max_games"
    schedule = _evaluation_schedule(games, seed)
    for g, (candidate_color, game_seed) in enumerate(schedule):
        game = play_game(candidate_color, game_seed)
        outcome = game["outcome"]
        if outcome not in ("win", "loss", "draw", "no_result"):
            raise ValueError("invalid evaluation outcome")
        wins += outcome == "win"
        losses += outcome == "loss"
        draws += outcome == "draw"
        no_results += outcome == "no_result"
        forced_scored += int(game["forced_scored"])
        played = g + 1
        points = wins + 0.5 * (draws + no_results)
        if gating:
            decision = gatekeeper_decision(points, played, games, required_candidate_win_prop)
        report({"event": "evaluation_game", "game": played, "games": games,
                "wins": wins, "losses": losses, "draws": draws, "no_results": no_results,
                "truncated": forced_scored, "forced_scored": forced_scored,
                "candidate_points": points, "baseline_points": played - points,
                "target_points": target, "decision": decision,
                "candidate_color": "black" if candidate_color == 1 else "white",
                "game_seed": game_seed, "moves": game["moves"], "outcome": outcome,
                "game_forced_scored": game["forced_scored"]})
        if gating and decision != "pending" and played < games:
            stop_reason = ("candidate_reached_target" if decision == "accepted"
                           else "candidate_cannot_reach_target")
            report({"event": "evaluation_stopped", "game": played, "games": games,
                    "decision": decision, "reason": stop_reason,
                    "candidate_points": points, "target_points": target})
            break
    rate = points / played
    interval = wilson_interval(points, played)
    elo = 400*math.log10(rate/(1-rate)) if 0 < rate < 1 else None
    return {"games": games, "played": played, "completed": played,
            "wins": wins, "losses": losses, "draws": draws, "no_results": no_results,
            "truncated": forced_scored, "forced_scored": forced_scored,
            "candidate_points": points, "baseline_points": played - points,
            "winrate": rate, "confidence95": interval, "elo_difference": elo,
            "seed": seed, "play_settings": settings,
            "early_stopped": played < games, "stop_reason": stop_reason,
            "gating": {"protocol": GATEKEEPER_PROTOCOL,
                       "required_candidate_win_prop": required_candidate_win_prop,
                       "target_points": target, "decision": decision} if gating else None}


def evaluate(config, candidate, baseline, games=DEFAULT_GATING_GAMES, report=print, score_utility=0.0,
             *, required_candidate_win_prop=None, seed=None, max_moves=1600):
    """Play a full measurement match, or a gatekeeper match with early stopping.

    Colors are shuffled in pairs, like KataGo MatchPairer. Games start empty
    and use randomized visit-based selection without root Dirichlet noise.
    Board size/rules/komi and the search engine remain this project's config.
    Set required_candidate_win_prop to enable gatekeeper decisions. Wilson
    intervals and Elo are descriptive only, especially after early stopping.
    """
    seed = _validate_evaluation_args(config, games, score_utility,
                                     required_candidate_win_prop, seed, max_moves)
    return _finish_evaluation(
        config, games, seed, required_candidate_win_prop, score_utility, max_moves, report,
        lambda candidate_color, game_seed: _play_evaluation_game(
            config, candidate, baseline, candidate_color, game_seed, score_utility, max_moves),
    )


def evaluate_parallel(config, candidate_path, baseline_path, games=DEFAULT_GATING_GAMES,
                      workers=2, report=print, score_utility=0.0, *,
                      required_candidate_win_prop=None, seed=None, max_moves=1600,
                      device=None):
    """Evaluate in spawned processes while aggregating in deterministic order.

    At most ``workers`` games are in flight at once.  A gatekeeper decision is
    checked after each result in schedule order; already-running games in the
    current batch may finish, but are discarded after an early decision and do
    not affect ``played`` or promotion.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 2:
        raise ValueError("parallel evaluation requires workers >= 2")
    seed = _validate_evaluation_args(config, games, score_utility,
                                     required_candidate_win_prop, seed, max_moves)
    schedule = _evaluation_schedule(games, seed)
    worker_count = min(games, workers)
    gating = required_candidate_win_prop is not None
    target = games * required_candidate_win_prop if gating else None
    settings = _play_settings(config, max_moves)
    settings["workers"] = worker_count
    settings["scheduler"] = "spawn-process-batches"
    report({"event": "evaluation_start", "games": games, "seed": seed,
            "protocol": GATEKEEPER_PROTOCOL if gating else "full-match-v2",
            "required_candidate_win_prop": required_candidate_win_prop,
            "target_points": target, "score_utility": score_utility,
            "play_settings": settings})
    wins = losses = draws = no_results = forced_scored = 0
    decision = "pending" if gating else None
    stop_reason = "max_games"
    played = 0
    points = 0.0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_evaluation_worker,
        initargs=(config.to_dict(), str(candidate_path), str(baseline_path),
                  None if device is None else str(device)),
    ) as pool:
        for start in range(0, games, worker_count):
            batch = schedule[start:start + worker_count]
            futures = [pool.submit(_evaluation_game_task, color, game_seed,
                                   score_utility, max_moves)
                       for color, game_seed in batch]
            for offset, future in enumerate(futures):
                candidate_color, game_seed = batch[offset]
                game = future.result()
                outcome = game["outcome"]
                if outcome not in ("win", "loss", "draw", "no_result"):
                    raise ValueError("invalid evaluation outcome")
                wins += outcome == "win"
                losses += outcome == "loss"
                draws += outcome == "draw"
                no_results += outcome == "no_result"
                forced_scored += int(game["forced_scored"])
                played += 1
                points = wins + 0.5 * (draws + no_results)
                if gating:
                    decision = gatekeeper_decision(points, played, games,
                                                   required_candidate_win_prop)
                report({"event": "evaluation_game", "game": played, "games": games,
                        "wins": wins, "losses": losses, "draws": draws,
                        "no_results": no_results, "truncated": forced_scored,
                        "forced_scored": forced_scored, "candidate_points": points,
                        "baseline_points": played - points, "target_points": target,
                        "decision": decision,
                        "candidate_color": "black" if candidate_color == 1 else "white",
                        "game_seed": game_seed, "moves": game["moves"],
                        "outcome": outcome, "game_forced_scored": game["forced_scored"]})
                if gating and decision != "pending" and played < games:
                    stop_reason = ("candidate_reached_target" if decision == "accepted"
                                   else "candidate_cannot_reach_target")
                    report({"event": "evaluation_stopped", "game": played, "games": games,
                            "decision": decision, "reason": stop_reason,
                            "candidate_points": points, "target_points": target})
                    for pending in futures[offset + 1:]:
                        pending.cancel()
                    break
            if gating and decision != "pending":
                break
    rate = points / played
    interval = wilson_interval(points, played)
    elo = 400*math.log10(rate/(1-rate)) if 0 < rate < 1 else None
    return {"games": games, "played": played, "completed": played,
            "wins": wins, "losses": losses, "draws": draws, "no_results": no_results,
            "truncated": forced_scored, "forced_scored": forced_scored,
            "candidate_points": points, "baseline_points": played - points,
            "winrate": rate, "confidence95": interval, "elo_difference": elo,
            "seed": seed, "play_settings": settings,
            "early_stopped": played < games, "stop_reason": stop_reason,
            "gating": {"protocol": GATEKEEPER_PROTOCOL,
                       "required_candidate_win_prop": required_candidate_win_prop,
                       "target_points": target, "decision": decision} if gating else None}

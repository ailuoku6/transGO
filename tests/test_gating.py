import json
import sys
from concurrent.futures import Future

import numpy as np
import pytest
import torch
import yaml

import transformer_go.evaluate as evaluation
import transformer_go.pipeline as pipeline
from transformer_go._native import Board
from transformer_go.checkpoint import load_checkpoint
from transformer_go.cli import main
from transformer_go.config import load_config
from transformer_go.gating import gatekeeper_decision
from transformer_go.runtime import seed_all


@pytest.mark.parametrize(
    "points,tallied,threshold,expected",
    [(24, 24, .5, "pending"), (25, 25, .5, "accepted"),
     (24, 49, .5, "pending"), (24, 50, .5, "rejected"),
     (25, 50, .5, "accepted"), (0, 25, .5, "pending"),
     (0, 26, .5, "rejected"), (24.5, 49, .5, "pending"),
     (25, 49, .5, "accepted"), (27.5, 50, .55, "accepted"),
     (27, 50, .55, "rejected"), (30, 30, .6, "accepted"),
     (0, 0, 0, "pending"), (0, 1, 0, "accepted"),
     (49, 49, 1, "pending"), (48, 49, 1, "rejected")],
)
def test_gatekeeper_boundaries_match_upstream(points, tallied, threshold, expected):
    assert gatekeeper_decision(points, tallied, 50, threshold) == expected


def test_early_decision_cannot_change_for_any_remaining_results():
    # An early decision must agree with *every* completion of the budget;
    # reaching 50% of the games played so far is insufficient.
    total = 10
    for threshold in (0, .5, .55, .7, 1):
        for played in range(1, total + 1):
            for half_points in range(played * 2 + 1):
                points = half_points / 2
                decision = gatekeeper_decision(points, played, total, threshold)
                possibilities = {
                    points + extra / 2 + 1e-10 >= total * threshold
                    for extra in range((total - played) * 2 + 1)
                }
                if decision == "accepted":
                    assert possibilities == {True}
                elif decision == "rejected":
                    assert possibilities == {False}
                else:
                    assert possibilities == {True, False}


def fake_games(monkeypatch, outcomes):
    remaining = iter(outcomes)
    calls = []

    def play(config, candidate, baseline, candidate_color, seed, score_utility, max_moves):
        calls.append((candidate_color, seed, score_utility))
        return {"outcome": next(remaining), "forced_scored": False, "moves": 90}

    monkeypatch.setattr(evaluation, "_play_evaluation_game", play)
    return calls


def test_gatekeeper_accepts_at_25_points_without_finishing_50_games(monkeypatch):
    calls = fake_games(monkeypatch, ["win"] * 25)
    events = []
    result = evaluation.evaluate(load_config("configs/smoke.yaml"), None, None, 50, events.append,
                                 required_candidate_win_prop=.5, score_utility=.05)
    assert len(calls) == result["played"] == result["completed"] == 25
    assert result["games"] == 50 and result["candidate_points"] == 25
    assert result["early_stopped"] and result["gating"]["decision"] == "accepted"
    assert result["stop_reason"] == "candidate_reached_target"
    assert pipeline.should_promote(result)
    assert events[0]["event"] == "evaluation_start"
    assert events[-1]["event"] == "evaluation_stopped"
    assert events[-1]["target_points"] == 25
    assert all(c[2] == .05 for c in calls)


def test_gatekeeper_rejects_at_26_losses(monkeypatch):
    calls = fake_games(monkeypatch, ["loss"] * 26)
    result = evaluation.evaluate(load_config("configs/smoke.yaml"), None, None, 50, lambda _: None,
                                 required_candidate_win_prop=.5)
    assert len(calls) == result["played"] == 26
    assert result["gating"]["decision"] == "rejected"
    assert result["stop_reason"] == "candidate_cannot_reach_target"
    assert not pipeline.should_promote(result)


def test_full_budget_tie_and_half_points_for_draws_and_no_results(monkeypatch):
    fake_games(monkeypatch, ["draw", "no_result"] * 25)
    result = evaluation.evaluate(load_config("configs/smoke.yaml"), None, None, 50, lambda _: None,
                                 required_candidate_win_prop=.5)
    assert result["played"] == 50 and not result["early_stopped"]
    assert result["draws"] == result["no_results"] == 25
    assert result["wins"] == result["losses"] == result["truncated"] == 0
    assert result["candidate_points"] == result["baseline_points"] == 25
    assert result["winrate"] == .5 and result["confidence95"][0] < .5
    assert result["gating"]["decision"] == "accepted"
    assert pipeline.should_promote(result)


def test_custom_threshold_requires_full_budget_points(monkeypatch):
    fake_games(monkeypatch, ["win"] * 28)
    result = evaluation.evaluate(load_config("configs/smoke.yaml"), None, None, 50, lambda _: None,
                                 required_candidate_win_prop=.55)
    assert result["played"] == 28 and result["gating"]["target_points"] == pytest.approx(27.5)
    assert pipeline.should_promote(result, .55)
    assert not pipeline.should_promote(result, .6)


def test_standalone_measurement_plays_full_budget_with_balanced_colors(monkeypatch):
    calls = fake_games(monkeypatch, ["win"] * 50)
    result = evaluation.evaluate(load_config("configs/smoke.yaml"), None, None, 50, lambda _: None)
    assert result["played"] == 50 and result["gating"] is None
    assert not result["early_stopped"]
    assert sum(color == 1 for color, _, _ in calls) == 25
    assert all(calls[g][0] == -calls[g + 1][0] for g in range(0, 50, 2))
    assert len({seed for _, seed, _ in calls}) == 50


def test_match_seed_is_reproducible_and_can_change_openings(monkeypatch):
    config = load_config("configs/smoke.yaml")
    traces = []
    for seed in (123, 123, 124):
        calls = fake_games(monkeypatch, ["draw"] * 4)
        evaluation.evaluate(config, None, None, 4, lambda _: None, seed=seed)
        traces.append(calls)
    assert traces[0] == traces[1] and traces[0] != traces[2]


def test_visit_temperature_starts_empty_and_does_not_restrict_pass(monkeypatch):
    config = load_config("configs/smoke.yaml")
    candidate, baseline = object(), object()
    seen = []

    def pass_search(board, evaluator, search_config, **kwargs):
        seen.append((board.move_number, evaluator, kwargs))
        visits = np.zeros(board.size ** 2 + 1)
        visits[-1] = 8
        return {"visits": visits}

    monkeypatch.setattr(evaluation, "search", pass_search)
    result = evaluation._play_evaluation_game(config, candidate, baseline, 1, 123, .05, 1600)
    assert [m for m, _, _ in seen] == [0, 1]
    assert [ev for _, ev, _ in seen] == [candidate, baseline]
    assert all(kw["noise"] is False and kw["min_moves"] == 0 for _, _, kw in seen)
    assert all(kw["score_utility"] == .05 for _, _, kw in seen)
    assert result == {"outcome": "loss", "forced_scored": False, "moves": 2}
    assert evaluation.gatekeeper_temperature(0, 9) == pytest.approx(.5)
    assert evaluation.gatekeeper_temperature(9, 9) == pytest.approx(.35)
    assert evaluation.gatekeeper_temperature(19, 19) == pytest.approx(.35)
    assert evaluation.gatekeeper_temperature(1000, 9) == pytest.approx(.2)


def test_move_limit_uses_katago_forced_score_and_keeps_game_counted(monkeypatch):
    config = load_config("configs/smoke.yaml")

    def one_stone(board, evaluator, search_config, **kwargs):
        visits = np.zeros(board.size ** 2 + 1)
        visits[0] = 8
        return {"visits": visits}

    monkeypatch.setattr(evaluation, "search", one_stone)
    # One move leaves the actual board nonterminal. Board.result() uses a
    # copy of BoardHistory to force score it, as gatekeeper does at its cap.
    assert not Board(3, .5).play(0).terminal()
    events = []
    result = evaluation.evaluate(config, None, None, 2, events.append, max_moves=1,
                                 required_candidate_win_prop=.75)
    assert result["wins"] == result["losses"] == 1
    assert result["truncated"] == result["forced_scored"] == result["completed"] == 2
    assert not pipeline.should_promote(result, .75)
    assert [e["forced_scored"] for e in events if e["event"] == "evaluation_game"] == [1, 2]


@pytest.mark.parametrize("games,threshold", [(0, .5), (3, .5), (50, -.1), (50, 1.1),
                                                (50, float("nan")), (50, float("inf"))])
def test_invalid_gate_options_fail_before_creating_run(tmp_path, games, threshold):
    directory = tmp_path / "run"
    with pytest.raises(ValueError):
        pipeline.run_pipeline(load_config("configs/smoke.yaml"), torch.device("cpu"), directory,
                              eval_games=games, promote_threshold=threshold, report=lambda _: None)
    assert not directory.exists()


def test_cli_resume_keeps_learner_and_best_uses_accepted_candidate(tmp_path, monkeypatch):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config.search.dirichlet_epsilon = 0
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config.to_dict()))
    directory = tmp_path / "run"

    def pass_only(boards):
        policy = np.zeros((len(boards), 10), dtype=np.float32)
        policy[:, -1] = 1
        return policy, np.zeros(len(boards), dtype=np.float32), np.zeros(len(boards), dtype=np.float32)

    monkeypatch.setattr(pipeline, "Evaluator", lambda *_: pass_only)
    calls = fake_games(monkeypatch, ["loss"] * 3 + ["win"] * 2 + ["loss"] * 3)
    args = ["transformer-go", "loop", "--config", str(cfg_path), "--output", str(directory),
            "--iterations", "1", "--games", "1", "--steps", "1", "--eval-games", "4",
            "--required-candidate-win-prop", "0.5"]
    for i in range(3):
        monkeypatch.setattr(sys, "argv", args)
        main()
    records = [json.loads((directory / f"iteration-{i:04d}.json").read_text()) for i in (1, 2, 3)]
    assert [d["promoted"] for d in records] == [False, True, False]
    assert [d["evaluation"]["played"] for d in records] == [3, 2, 3]
    assert len({d["eval_seed"] for d in records}) == 3
    assert all(d["promotion_policy"] == "katago-gatekeeper-v1" for d in records)
    assert all(d["required_candidate_win_prop"] == .5 for d in records)
    assert len(calls) == 8
    best_hash = records[1]["candidate_sha256"]
    assert records[0]["baseline_sha256"] == records[0]["best_sha256"]
    assert records[1]["baseline_sha256"] == records[0]["best_sha256"]
    assert records[1]["best_sha256"] == records[2]["baseline_sha256"] == records[2]["best_sha256"] == best_hash
    for name, step in (("best.pt", 2), ("learner.pt", 3)):
        _, saved, payload = load_checkpoint(directory / name, torch.device("cpu"), selfplay_only=True)
        assert saved.to_dict() == config.to_dict() and payload["step"] == step
    # Self-play always uses the accepted model's hash, including after resume.
    from transformer_go.data import read_shard
    replay_models = {read_shard(p)[1]["model_version"] for p in (directory / "replay").glob("*.npz")}
    assert replay_models == {records[0]["baseline_sha256"], best_hash}


def test_cli_defaults_are_200_games_and_half_points(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    monkeypatch.setattr(sys, "argv", ["transformer-go", "loop", "--config", str(path)])
    main()
    assert captured["args"][6] == 200
    assert captured["kwargs"]["promote_threshold"] == .5


def test_cli_loop_forwards_process_worker_counts(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(pipeline, "run_pipeline",
                        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    monkeypatch.setattr(sys, "argv", [
        "transformer-go", "loop", "--config", str(path),
        "--selfplay-workers", "2", "--eval-workers", "3"])
    main()
    assert captured["kwargs"]["selfplay_workers"] == 2
    assert captured["kwargs"]["eval_workers"] == 3


def test_parallel_evaluation_batches_and_preserves_gatekeeper_accounting(monkeypatch):
    config = load_config("configs/smoke.yaml")
    calls = []

    class ImmediatePool:
        def __init__(self, *args, initializer=None, initargs=(), **kwargs):
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, function, *args):
            future = Future()
            calls.append(args)
            future.set_result(function(*args))
            return future

    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(evaluation, "_init_evaluation_worker", lambda *args: None)

    def game(color, seed, score_utility, max_moves):
        return {"outcome": "win", "forced_scored": False, "moves": 2}

    monkeypatch.setattr(evaluation, "_evaluation_game_task", game)
    events = []
    result = evaluation.evaluate_parallel(
        config, "candidate.pt", "baseline.pt", games=2, workers=2, report=events.append,
        required_candidate_win_prop=.5, seed=123, device=torch.device("cpu"))
    assert result["played"] == 1
    assert result["gating"]["decision"] == "accepted"
    assert result["play_settings"]["workers"] == 2
    assert [event["event"] for event in events] == [
        "evaluation_start", "evaluation_game", "evaluation_stopped"]
    # The second game was submitted in the already-running batch, but it was
    # not tallied after the first result made the gatekeeper decision.
    assert len(calls) == 2

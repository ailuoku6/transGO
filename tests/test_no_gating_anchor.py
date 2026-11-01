import json
import sys

import numpy as np
import pytest
import torch
import yaml

import transformer_go.evaluate as evaluation
import transformer_go.pipeline as pipeline
from transformer_go.checkpoint import load_checkpoint, model_version, save_checkpoint
from transformer_go.cli import main
from transformer_go.config import load_config
from transformer_go.model import PolicyValueNet


def pass_only(boards):
    policy = np.zeros((len(boards), 10), dtype=np.float32)
    policy[:, -1] = 1
    return policy, np.zeros(len(boards), dtype=np.float32), np.zeros(len(boards), dtype=np.float32)


def smoke_config(tmp_path):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config.search.dirichlet_epsilon = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    return config, path


def record_games(monkeypatch, outcomes):
    remaining = iter(outcomes)
    calls = []

    def play(config, candidate, baseline, candidate_color, seed, score_utility, max_moves, temperature=None):
        calls.append(seed)
        outcome = next(remaining)
        return {"outcome": outcome, "forced_scored": False, "moves": 2,
                "actions": [config.board_size ** 2] * 2,
                "score": candidate_color * (1 if outcome == "win" else -1),
                "move_comments": ["test fixture"] * 2}

    monkeypatch.setattr(evaluation, "_play_evaluation_game", play)
    return calls


def test_no_gating_promotes_every_candidate_without_a_match(tmp_path, monkeypatch):
    config, cfg_path = smoke_config(tmp_path)
    directory = tmp_path / "run"
    monkeypatch.setattr(pipeline, "Evaluator", lambda *_, **__: pass_only)
    calls = record_games(monkeypatch, [])
    args = ["transformer-go", "loop", "--config", str(cfg_path), "--output", str(directory),
            "--iterations", "2", "--games", "1", "--steps", "1", "--no-gating"]
    monkeypatch.setattr(sys, "argv", args)
    main()
    records = [json.loads((directory / f"iteration-{i:04d}.json").read_text()) for i in (1, 2)]
    assert calls == []
    assert [r["promoted"] for r in records] == [True, True]
    assert all(r["promotion_policy"] == "always-promote-v1" for r in records)
    assert all(r["evaluation"] is None and r["eval_seed"] is None for r in records)
    assert all(r["required_candidate_win_prop"] is None for r in records)
    assert all(r["best_sha256"] == r["candidate_sha256"] for r in records)
    assert records[1]["baseline_sha256"] == records[0]["candidate_sha256"]
    assert not (directory / "evaluation").exists()
    for name, step in (("best.pt", 2), ("learner.pt", 2)):
        _, saved, payload = load_checkpoint(directory / name, torch.device("cpu"), selfplay_only=True)
        assert saved.to_dict() == config.to_dict() and payload["step"] == step


def test_anchor_check_runs_on_schedule_and_never_changes_promotion(tmp_path, monkeypatch):
    config, cfg_path = smoke_config(tmp_path)
    directory = tmp_path / "run"
    anchor = tmp_path / "anchor.pt"
    save_checkpoint(anchor, PolicyValueNet(config.model), config)
    anchor_hash = model_version(anchor)
    monkeypatch.setattr(pipeline, "Evaluator", lambda *_, **__: pass_only)
    # Iteration 1: gatekeeper loses 2 games. Iteration 2: gatekeeper loses 2,
    # then the anchor match plays all 4 games despite an early 2-0 lead.
    calls = record_games(monkeypatch, ["loss"] * 4 + ["win", "win", "loss", "win"])
    args = ["transformer-go", "loop", "--config", str(cfg_path), "--output", str(directory),
            "--iterations", "2", "--games", "1", "--steps", "1", "--eval-games", "2",
            "--required-candidate-win-prop", "0.5",
            "--anchor-checkpoint", str(anchor), "--anchor-every", "2", "--anchor-games", "4"]
    monkeypatch.setattr(sys, "argv", args)
    main()
    records = [json.loads((directory / f"iteration-{i:04d}.json").read_text()) for i in (1, 2)]
    assert len(calls) == 8
    assert [r["promoted"] for r in records] == [False, False]
    assert "anchor_evaluation" not in records[0]
    check = records[1]["anchor_evaluation"]
    assert check["protocol"] == "fixed-anchor-v1"
    assert check["anchor_sha256"] == anchor_hash
    assert check["subject_sha256"] == records[1]["best_sha256"]
    assert check["seed"] == pipeline.anchor_match_seed(config.seed, 2)
    assert check["result"]["played"] == 4 and check["result"]["wins"] == 3
    assert check["result"]["gating"] is None and not check["result"]["early_stopped"]
    snapshot = directory / "anchors" / f"anchor-{anchor_hash[:16]}.pt"
    assert check["anchor"] == str(snapshot) and model_version(snapshot) == anchor_hash
    assert len(list((directory / "evaluation" / "anchor-0002").glob("*/*.sgf"))) == 4
    lines = (directory / "anchor-evaluations.jsonl").read_text().splitlines()
    assert [json.loads(line)["iteration"] for line in lines] == [2]
    assert json.loads(lines[0])["winrate"] == .75


@pytest.mark.parametrize("kwargs,message", [
    ({"anchor_every": 2}, "requires anchor_checkpoint"),
    ({"anchor_every": -1}, "nonnegative"),
    ({"anchor_every": 1, "anchor_checkpoint": "missing.pt"}, "does not exist"),
])
def test_invalid_anchor_options_fail_before_creating_run(tmp_path, kwargs, message):
    directory = tmp_path / "run"
    with pytest.raises(ValueError, match=message):
        pipeline.run_pipeline(load_config("configs/smoke.yaml"), torch.device("cpu"), directory,
                              eval_games=2, report=lambda _: None, **kwargs)
    assert not directory.exists()


def test_anchor_with_other_architecture_is_rejected(tmp_path):
    config = load_config("configs/smoke.yaml")
    other = load_config("configs/smoke.yaml")
    other.model.width = 16
    anchor = tmp_path / "anchor.pt"
    save_checkpoint(anchor, PolicyValueNet(other.model), other)
    with pytest.raises(ValueError, match="architecture"):
        pipeline.run_pipeline(config, torch.device("cpu"), tmp_path / "run", eval_games=2,
                              report=lambda _: None, anchor_checkpoint=anchor, anchor_every=1)

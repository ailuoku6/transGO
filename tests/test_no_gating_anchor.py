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
    assert check["protocol"] == "strongest-confirmed-anchor-v1"
    assert check["anchor_sha256"] == anchor_hash
    assert check["subject_sha256"] == records[1]["best_sha256"]
    assert check["seed"] == pipeline.anchor_match_seed(config.seed, 2)
    assert check["result"]["played"] == 4 and check["result"]["wins"] == 3
    assert check["result"]["gating"] is None and not check["result"]["early_stopped"]
    assert check["anchor_advanced"] is True
    assert check["active_anchor_sha256"] == check["subject_sha256"]
    snapshot = directory / "anchors" / f"anchor-{anchor_hash[:16]}.pt"
    assert check["anchor"] == str(snapshot) and model_version(snapshot) == anchor_hash
    assert len(list((directory / "evaluation" / "anchor-0002").glob("*/*.sgf"))) == 4
    lines = (directory / "anchor-evaluations.jsonl").read_text().splitlines()
    assert [json.loads(line)["iteration"] for line in lines] == [2]
    assert json.loads(lines[0])["winrate"] == .75


def test_anchor_advances_only_after_win_and_survives_resume(tmp_path, monkeypatch):
    config, cfg_path = smoke_config(tmp_path)
    directory = tmp_path / "run"
    anchor = tmp_path / "anchor.pt"
    save_checkpoint(anchor, PolicyValueNet(config.model), config)
    seed_hash = model_version(anchor)
    monkeypatch.setattr(pipeline, "Evaluator", lambda *_, **__: pass_only)
    calls = record_games(monkeypatch, ["win"] * 4 + ["win"] * 2 + ["loss"] * 6 + ["win"] * 4)
    args = ["transformer-go", "loop", "--config", str(cfg_path), "--output", str(directory),
            "--games", "1", "--steps", "1", "--no-gating", "--anchor-checkpoint", str(anchor),
            "--anchor-every", "1", "--anchor-games", "4"]
    monkeypatch.setattr(sys, "argv", args + ["--iterations", "1"])
    main()
    first = json.loads((directory / "iteration-0001.json").read_text())
    first_check = first["anchor_evaluation"]
    assert first_check["anchor_sha256"] == seed_hash
    assert first_check["anchor_advanced"] is True
    champion = first_check["subject_sha256"]
    assert champion != seed_hash
    # The source is only an initial seed. A resumed run keeps its confirmed snapshot.
    anchor.unlink()
    assert not anchor.exists()
    monkeypatch.setattr(sys, "argv", args + ["--iterations", "3"])
    main()
    checks = [json.loads((directory / f"iteration-{i:04d}.json").read_text())["anchor_evaluation"]
              for i in (1, 2, 3, 4)]
    assert len(calls) == 16
    assert [c["anchor_sha256"] for c in checks] == [seed_hash, champion, champion, champion]
    assert [c["anchor_advanced"] for c in checks] == [True, False, False, True]
    assert checks[1]["result"]["winrate"] == 0.5
    assert checks[2]["result"]["winrate"] == 0.0
    active = json.loads((directory / "anchors" / "active-anchor.json").read_text())
    assert active["sha256"] == checks[3]["subject_sha256"]
    assert active["iteration"] == 4
    assert model_version(directory / "anchors" / active["filename"]) == active["sha256"]


def test_anchor_identity_compares_weights_not_training_metadata(tmp_path):
    config, _ = smoke_config(tmp_path)
    model = PolicyValueNet(config.model)
    initial, metadata_only, changed = [tmp_path / f"{name}.pt"
                                       for name in ("initial", "metadata", "changed")]
    save_checkpoint(initial, model, config, step=0)
    save_checkpoint(metadata_only, model, config, step=1)
    assert model_version(initial) != model_version(metadata_only)
    assert not pipeline._models_differ(initial, metadata_only)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    save_checkpoint(changed, model, config, step=1)
    assert pipeline._models_differ(initial, changed)


def test_omitted_anchor_checkpoint_uses_initial_best(tmp_path, monkeypatch):
    config, cfg_path = smoke_config(tmp_path)
    directory = tmp_path / "run"
    monkeypatch.setattr(pipeline, "Evaluator", lambda *_, **__: pass_only)
    record_games(monkeypatch, ["win"] * 4)
    monkeypatch.setattr(sys, "argv", [
        "transformer-go", "loop", "--config", str(cfg_path), "--output", str(directory),
        "--iterations", "1", "--games", "1", "--steps", "1", "--no-gating",
        "--anchor-every", "1", "--anchor-games", "4"])
    main()
    record = json.loads((directory / "iteration-0001.json").read_text())
    check = record["anchor_evaluation"]
    assert check["anchor_sha256"] == record["baseline_sha256"]
    initial_snapshot = directory / "anchors" / f"anchor-{record['baseline_sha256'][:16]}.pt"
    assert model_version(initial_snapshot) == record["baseline_sha256"]


@pytest.mark.parametrize("kwargs,message", [
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

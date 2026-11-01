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
from transformer_go.regression_guard import MAX_ROLLBACKS_PER_RESTORE, apply_regression_guard


def write_weights(path, step, seed):
    config = load_config("configs/smoke.yaml")
    torch.manual_seed(seed)
    save_checkpoint(path, PolicyValueNet(config.model), config, step=step,
                    rng=np.random.default_rng(seed))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    best, learner = root / "best.pt", root / "learner.pt"
    write_weights(best, 10, 1)
    write_weights(learner, 12, 2)
    return root, best, learner


def guard(run, iteration, winrate, anchor="a" * 64, drop=.15):
    root, best, learner = run
    return apply_regression_guard(root, iteration, best, learner, anchor, winrate, drop,
                                  report=lambda _: None)


def state(root):
    return json.loads((root / "anchors" / "regression-guard.json").read_text())


def test_first_check_and_new_high_store_restore_points(run):
    root, best, learner = run
    first = guard(run, 25, .5)
    assert first["decision"] == "new_restore_point"
    point = first["restore_point"]
    assert point["best_sha256"] == model_version(best)
    assert point["learner_sha256"] == model_version(learner)
    assert point["learner_step"] == 12
    write_weights(best, 20, 3)
    write_weights(learner, 22, 4)
    second = guard(run, 50, .6)
    assert second["decision"] == "new_restore_point"
    assert second["replaced_restore_point"]["iteration"] == 25
    names = sorted(p.name for p in (root / "anchors").glob("restore-*.pt"))
    assert names == ["restore-0050-best.pt", "restore-0050-learner.pt"]
    assert state(root)["restore_point"]["iteration"] == 50


def test_small_drop_holds_without_touching_weights(run):
    root, best, learner = run
    guard(run, 25, .6)
    write_weights(best, 20, 3)
    before = model_version(best), model_version(learner)
    record = guard(run, 50, .46)
    assert record["decision"] == "hold"
    assert record["drop"] == pytest.approx(.14)
    assert (model_version(best), model_version(learner)) == before


def test_large_drop_restores_best_and_learner_and_keeps_step_increasing(run):
    root, best, learner = run
    saved = guard(run, 25, .6)["restore_point"]
    write_weights(best, 30, 5)
    write_weights(learner, 32, 6)
    _, _, current = load_checkpoint(learner, torch.device("cpu"))
    record = guard(run, 50, .45)
    assert record["decision"] == "rollback"
    assert record["drop"] == pytest.approx(.15)
    assert record["before"]["learner_step"] == 32
    assert record["after"]["learner_step"] == 32
    assert record["after"]["restored_learner_step"] == 12
    assert model_version(best) == saved["best_sha256"] == record["after"]["best_sha256"]
    restored, _, payload = load_checkpoint(learner, torch.device("cpu"))
    original, _, _ = load_checkpoint(saved["learner_path"], torch.device("cpu"))
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
    assert payload["step"] == 32
    assert payload["numpy_rng"] == current["numpy_rng"]
    assert state(root)["rollbacks_to_restore_point"] == 1


def test_repeated_rollbacks_stop_and_rebase_on_current_result(run):
    root, best, learner = run
    guard(run, 25, .8)
    for i in range(MAX_ROLLBACKS_PER_RESTORE):
        write_weights(best, 40 + i, 10 + i)
        assert guard(run, 50 + 25 * i, .5)["decision"] == "rollback"
    write_weights(best, 90, 20)
    current = model_version(best)
    record = guard(run, 100, .5)
    assert record["decision"] == "rebase_after_rollback_limit"
    assert record["restore_point"]["iteration"] == 100
    assert record["restore_point"]["best_sha256"] == current == model_version(best)
    assert state(root)["rollbacks_to_restore_point"] == 0
    assert not (root / "anchors" / "restore-0025-best.pt").exists()


def test_changed_anchor_or_missing_restore_point_resets_reference(run):
    root, best, learner = run
    guard(run, 25, .9)
    record = guard(run, 50, .2, anchor="b" * 64)
    assert record["decision"] == "new_restore_point" and record["reset_reason"] == "anchor_changed"
    assert not (root / "anchors" / "restore-0025-best.pt").exists()
    (root / "anchors" / "restore-0050-best.pt").unlink()
    record = guard(run, 75, .1, anchor="b" * 64)
    assert record["decision"] == "new_restore_point"
    assert record["reset_reason"] == "restore_point_missing"


def test_loop_records_rollback_and_selfplay_uses_restored_best(tmp_path, monkeypatch):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config.search.dirichlet_epsilon = 0
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config.to_dict()))
    anchor = tmp_path / "anchor.pt"
    save_checkpoint(anchor, PolicyValueNet(config.model), config)
    directory = tmp_path / "run"

    def pass_only(boards):
        policy = np.zeros((len(boards), 10), dtype=np.float32)
        policy[:, -1] = 1
        return policy, np.zeros(len(boards), dtype=np.float32), np.zeros(len(boards), dtype=np.float32)

    outcomes = iter(["win"] * 4 + ["win"] + ["loss"] * 3 + ["win"] * 4)

    def play(config, candidate, baseline, candidate_color, seed, score_utility, max_moves, temperature=None):
        outcome = next(outcomes)
        return {"outcome": outcome, "forced_scored": False, "moves": 2,
                "actions": [config.board_size ** 2] * 2,
                "score": candidate_color * (1 if outcome == "win" else -1),
                "move_comments": ["test fixture"] * 2}

    monkeypatch.setattr(pipeline, "Evaluator", lambda *_, **__: pass_only)
    monkeypatch.setattr(evaluation, "_play_evaluation_game", play)
    monkeypatch.setattr(sys, "argv", [
        "transformer-go", "loop", "--config", str(cfg_path), "--output", str(directory),
        "--iterations", "3", "--games", "1", "--steps", "1", "--no-gating",
        "--anchor-checkpoint", str(anchor), "--anchor-every", "1", "--anchor-games", "4",
        "--rollback-drop", "0.3"])
    main()
    records = [json.loads((directory / f"iteration-{i:04d}.json").read_text()) for i in (1, 2, 3)]
    assert [r["regression_guard"]["decision"] for r in records] == [
        "new_restore_point", "rollback", "hold"]
    assert [r["rolled_back"] for r in records] == [False, True, False]
    rollback = records[1]
    assert rollback["regression_guard"]["drop"] == pytest.approx(.75)
    assert rollback["best_sha256_before_rollback"] == rollback["candidate_sha256"]
    assert rollback["best_sha256"] == records[0]["candidate_sha256"]
    assert rollback["regression_guard"]["after"]["best_sha256"] == records[0]["candidate_sha256"]
    assert "best_sha256_before_rollback" not in records[2]
    history = [json.loads(line) for line in (directory / "regression-guard.jsonl").read_text().splitlines()]
    assert [h["decision"] for h in history] == ["new_restore_point", "rollback", "hold"]
    assert history[1]["reference_iteration"] == 1
    assert history[1]["restored_best_sha256"] == records[0]["candidate_sha256"]
    from transformer_go.data import read_shard
    versions = [read_shard(p)[1]["model_version"] for p in (directory / "replay").glob("*.npz")]
    assert versions.count(records[0]["candidate_sha256"]) == 2
    assert records[1]["candidate_sha256"] not in versions
    steps = [r["training"]["end_step"] for r in records]
    assert steps == [1, 2, 3]
    _, _, payload = load_checkpoint(directory / "learner.pt", torch.device("cpu"))
    assert payload["step"] == 3


@pytest.mark.parametrize("drop,every,message", [
    (.15, 0, "requires anchor_every"), (0, 5, r"\(0, 1\]"), (1.5, 5, r"\(0, 1\]"),
    (float("nan"), 5, r"\(0, 1\]")])
def test_invalid_rollback_options_fail_before_creating_run(tmp_path, drop, every, message):
    directory = tmp_path / "run"
    with pytest.raises(ValueError, match=message):
        pipeline.run_pipeline(load_config("configs/smoke.yaml"), torch.device("cpu"), directory,
                              eval_games=2, report=lambda _: None, anchor_every=every,
                              anchor_checkpoint=tmp_path / "missing.pt", rollback_drop=drop)
    assert not directory.exists()

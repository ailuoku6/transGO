import json
import sys

import numpy as np
import pytest
import torch
import yaml

from transformer_go._native import Board, Search
from transformer_go.checkpoint import load_checkpoint
from transformer_go.cli import main
from transformer_go.config import load_config
from transformer_go.data import read_shard, validate
from transformer_go.selfplay import generate, play_game


def pass_only(boards):
    policies = np.zeros((len(boards), boards[0].size**2+1), dtype=np.float32)
    policies[:, -1] = 1
    return policies, np.zeros(len(boards), dtype=np.float32)


@pytest.mark.parametrize("epsilon", [0.0, 0.25])
def test_pass_constraint_applies_inside_tree_and_to_root_noise(epsilon):
    seen = []
    def evaluator(boards):
        seen.extend(boards)
        return pass_only(boards)
    board = Board(5, 0.5)
    result = Search(seed=5).run(board, evaluator, 64, 4, epsilon=epsilon, min_moves=8)
    assert result["visits"][-1] == 0 and result["policy"][-1] == 0
    assert sum(result["visits"]) == 64 and np.isclose(sum(result["policy"]), 1)
    assert any(b.move_number >= 2 for b in seen)
    assert all(b.passes == 0 for b in seen if b.move_number <= 8)
    # Rules remain unchanged for actual play and unconstrained analysis/evaluation.
    assert board.legal_mask()[-1] and board.play(25).passes == 1
    assert Search().run(board, pass_only, 8, 4)["policy"][-1] > 0


def test_pass_restored_exactly_at_threshold():
    board = Board(5, 0.5).play(0).play(1)
    assert Search().run(board, pass_only, 8, 4, min_moves=3)["policy"][-1] == 0
    assert Search().run(board, pass_only, 8, 4, min_moves=2)["policy"][-1] > 0


def test_no_nonpass_legal_move_allows_forced_pass():
    # Black has two eyes at 1 and 7. White cannot play anywhere on the board.
    board = Board(3, 0.5)
    for action in [0, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 8]:
        board = board.play(action)
    assert not any(board.legal_mask()[:-1]) and not board.terminal()
    result = Search().run(board, pass_only, 8, 4, min_moves=20)
    assert result["policy"][-1] == 1 and result["simulations"] == 8


def test_selfplay_targets_and_metadata_respect_min_moves(tmp_path):
    config = load_config("configs/smoke.yaml")
    config.search.dirichlet_epsilon = 0
    # Even a network that always passes must explore before the threshold.
    default = play_game(config, pass_only, 42)
    assert default[1]["moves"] == 2
    paths = generate(config, pass_only, tmp_path, 1, report=lambda _: None, min_moves=4)
    data, meta = read_shard(paths[0])
    validate(data)
    assert meta["moves"] >= 6 and meta["min_moves"] == 4
    assert meta["early_forced_passes"] == 0
    assert not data["policy"][:4, -1].any()
    # The target legal mask is still the rules mask, so pass receives negative supervision.
    assert data["legal"][:4, -1].all()
    assert paths[0].with_suffix(".sgf").exists()


@pytest.mark.parametrize("min_moves", [-1, 135, 0.5, True])
def test_invalid_min_moves_rejected_before_writing_replay(tmp_path, min_moves):
    with pytest.raises(ValueError, match="min_moves"):
        generate(load_config("configs/smoke.yaml"), pass_only, tmp_path/"replay",
                 min_moves=min_moves)
    assert not (tmp_path/"replay").exists()


def test_native_rejects_negative_min_moves():
    with pytest.raises(ValueError):
        Search().run(Board(3), pass_only, min_moves=-1)


def test_loop_cli_resume_can_change_min_moves_without_changing_checkpoint(tmp_path, monkeypatch, capsys):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config_path = tmp_path/"config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    directory = tmp_path/"run"
    args = ["transformer-go", "loop", "--config", str(config_path), "--output", str(directory),
            "--iterations", "1", "--games", "1", "--steps", "1", "--eval-games", "2"]
    monkeypatch.setattr(sys, "argv", args)
    main()
    monkeypatch.setattr(sys, "argv", args+["--min-moves", "4"])
    main()
    record = json.loads((directory/"iteration-0002.json").read_text())
    assert record["min_moves"] == 4 and record["iteration"] == 2
    _, saved, payload = load_checkpoint(directory/"learner.pt", torch.device("cpu"))
    assert payload["step"] == 2 and saved.to_dict() == config.to_dict()
    assert sorted(read_shard(p)[1]["min_moves"] for p in (directory/"replay").glob("*.npz")) == [0, 4]
    from pathlib import Path
    from transformer_go.sgf import parse
    for iteration in (1, 2):
        saved_record = json.loads((directory / f"iteration-{iteration:04d}.json").read_text())
        match = saved_record["evaluation"]
        files = list(Path(match["sgf_dir"]).glob("*.sgf"))
        assert len(files) == match["played"]
        assert all(path.is_relative_to(directory / "evaluation") for path in files)
        for path in files:
            meta = json.loads(parse(path.read_text())[0]["C"][0])
            assert saved_record["candidate_sha256"] in meta["candidate"]
            assert saved_record["baseline_sha256"] in meta["baseline"]
            assert meta["play_settings"]["temperature_override"] is None
    assert '"min_moves": 4' in capsys.readouterr().out


def test_parallel_selfplay_cli_propagates_min_moves(tmp_path, monkeypatch, capsys):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config_path = tmp_path/"config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    replay = tmp_path/"replay"
    monkeypatch.setattr(sys, "argv", ["transformer-go", "selfplay", "--config", str(config_path),
                                     "--output", str(replay), "--games", "2", "--workers", "2",
                                     "--min-moves", "4", "--inference-batch-size", "8",
                                     "--inference-wait-ms", "0.5"])
    main()
    paths = list(replay.glob("*.npz"))
    assert len(paths) == 2
    for path in paths:
        data, meta = read_shard(path)
        assert meta["min_moves"] == 4 and meta["moves"] >= 6
        assert meta["tree_reuse"]
        assert not data["policy"][:4, -1].any()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = next(event for event in events if event["event"] == "inference_summary")
    assert summary["max_batch_size"] == 8 and summary["max_wait_ms"] == .5
    assert summary["models"]["selfplay"]["positions"] > 0

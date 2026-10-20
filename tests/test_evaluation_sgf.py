import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import transformer_go.evaluate as evaluation
from transformer_go._native import Board
from transformer_go.checkpoint import save_checkpoint
from transformer_go.cli import main
from transformer_go.config import load_config
from transformer_go.model import PolicyValueNet
from transformer_go.sgf import dumps, parse


def pass_search(board, evaluator, config, **kwargs):
    visits = np.zeros(board.size ** 2 + 1)
    visits[-1] = 8
    return {"visits": visits}


def replay_sgf(path):
    nodes = parse(path.read_text(encoding="utf-8"))
    root = nodes[0]
    size = int(root["SZ"][0])
    board = Board(size, float(root["KM"][0]), root["RU"][0])
    for node in nodes[1:]:
        coord = node["B" if board.to_play == 1 else "W"][0]
        action = size ** 2 if not coord else (ord(coord[1]) - 97) * size + ord(coord[0]) - 97
        board = board.play(action)
    return nodes, board


def test_serial_records_players_result_and_search_comments(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "search", pass_search)
    config = load_config("configs/smoke.yaml")
    events = []
    candidate, baseline = r"candidate]\棋手", r"baseline]\model"
    result = evaluation.evaluate(
        config, None, None, games=2, seed=41, report=events.append,
        sgf_dir=tmp_path, candidate_name=candidate, baseline_name=baseline)
    files = sorted(Path(result["sgf_dir"]).glob("*.sgf"))
    assert len(files) == result["played"] == 2
    games = [e for e in events if e["event"] == "evaluation_game"]
    assert events[0]["sgf_dir"] == result["sgf_dir"]
    for path, event in zip(files, games):
        nodes, board = replay_sgf(path)
        root = nodes[0]
        meta = json.loads(root["C"][0])
        assert root["RE"] == ["W+0.5"] and board.terminal()
        assert len(nodes) - 1 == event["moves"] == 2
        assert root["PB"] == [candidate if event["candidate_color"] == "black" else baseline]
        assert root["PW"] == [baseline if event["candidate_color"] == "black" else candidate]
        assert meta["game_seed"] == event["game_seed"]
        assert meta["outcome"] == event["outcome"]
        assert meta["play_settings"]["root_noise"] is False
        assert meta["termination"] == "normal"
        assert "MCTS top=pass" in nodes[1]["C"][0]
        assert "chosen is top=True" in nodes[1]["C"][0]
        assert event["sgf_path"] == str(path)


def test_recording_preserves_rng_outcomes_and_existing_matches(tmp_path, monkeypatch):
    seeds = []

    def varied_search(board, evaluator, config, **kwargs):
        seeds.append(kwargs["seed"])
        visits = np.asarray(board.legal_mask(), dtype=float)
        visits[-1] = 2
        return {"visits": visits}

    monkeypatch.setattr(evaluation, "search", varied_search)
    config = load_config("configs/smoke.yaml")

    def run(directory):
        return evaluation.evaluate(config, None, None, 2, lambda _: None,
                                   max_moves=7, seed=12, sgf_dir=directory)

    without = run(None)
    expected_seeds = list(seeds)
    seeds.clear()
    first = run(tmp_path)
    assert seeds == expected_seeds
    first_dir = first.pop("sgf_dir")
    assert first == without
    saved = {p.name: p.read_bytes() for p in Path(first_dir).glob("*.sgf")}
    second = run(tmp_path)
    assert second.pop("sgf_dir") != first_dir and second == without
    assert {p.name: p.read_bytes() for p in Path(first_dir).glob("*.sgf")} == saved


def test_forced_scoring_records_actual_moves_without_synthetic_passes(tmp_path, monkeypatch):
    def one_move(board, evaluator, config, **kwargs):
        visits = np.zeros(board.size ** 2 + 1)
        visits[0] = 8
        return {"visits": visits}

    monkeypatch.setattr(evaluation, "search", one_move)
    result = evaluation.evaluate(load_config("configs/smoke.yaml"), None, None,
                                 2, lambda _: None, max_moves=1, sgf_dir=tmp_path)
    for path in Path(result["sgf_dir"]).glob("*.sgf"):
        nodes, board = replay_sgf(path)
        assert len(nodes) == 2 and not board.terminal()
        assert nodes[1]["B"] == ["aa"]
        meta = json.loads(nodes[0]["C"][0])
        assert meta["forced_scored"] and meta["termination"] == "move_limit"
        assert nodes[0]["RE"] == [f"B+{board.score():g}"]


@pytest.mark.parametrize("outcome,score,result", [("draw", 0, "0"), ("no_result", None, "Void")])
def test_draw_and_no_result_export_distinct_results(tmp_path, monkeypatch, outcome, score, result):
    def play(*args):
        return {"outcome": outcome, "forced_scored": False, "moves": 2,
                "actions": [9, 9], "score": score, "move_comments": ["", ""]}

    monkeypatch.setattr(evaluation, "_play_evaluation_game", play)
    match = evaluation.evaluate(load_config("configs/smoke.yaml"), None, None,
                                2, lambda _: None, sgf_dir=tmp_path)
    for path in Path(match["sgf_dir"]).glob("*.sgf"):
        assert parse(path.read_text())[0]["RE"] == [result]
    assert match["candidate_points"] == 1


def test_zero_temperature_always_chooses_most_visited_move(monkeypatch):
    def tied_candidates(board, evaluator, config, **kwargs):
        visits = np.zeros(board.size ** 2 + 1)
        visits[0], visits[4] = 9, 10
        return {"visits": visits}

    monkeypatch.setattr(evaluation, "search", tied_candidates)
    config = load_config("configs/smoke.yaml")
    for seed in range(20):
        game = evaluation._play_evaluation_game(config, None, None, 1, seed, 0, 1, 0)
        assert game["actions"] == [4]
        assert "temperature=0; MCTS top=B2" in game["move_comments"][0]
        assert "chosen is top=True" in game["move_comments"][0]


@pytest.mark.parametrize("temperature,threshold", [(-1, None), (float("nan"), None),
                                                   (float("inf"), None), (0, .5)])
@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_temperature_fails_before_writing(tmp_path, temperature, threshold, parallel):
    function = evaluation.evaluate_parallel if parallel else evaluation.evaluate
    with pytest.raises(ValueError, match="temperature"):
        function(load_config("configs/smoke.yaml"), None, None, games=2,
                 sgf_dir=tmp_path / "records", temperature=temperature,
                 required_candidate_win_prop=threshold)
    assert not (tmp_path / "records").exists()


def test_sgf_escapes_root_and_move_comments():
    text = r"comment ] \ 中文"
    nodes = parse(dumps(9, 7.5, [0, 81], score=1.5, black_name=text,
                        white_name=text, comment=text, move_comments=[text, text]))
    assert nodes[0]["PB"] == nodes[0]["PW"] == nodes[0]["C"] == [text]
    assert nodes[1]["C"] == nodes[2]["C"] == [text]
    assert nodes[2]["W"] == [""]
    with pytest.raises(ValueError, match="comments"):
        dumps(9, 7.5, [0], move_comments=[])


@pytest.mark.parametrize("workers", [1, 2])
def test_eval_cli_exports_real_games_at_zero_temperature(tmp_path, monkeypatch, capsys, workers):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, PolicyValueNet(config.model), config)
    monkeypatch.setattr(sys, "argv", [
        "transformer-go", "eval", "--config", str(config_path),
        "--candidate", str(checkpoint), "--baseline", str(checkpoint),
        "--games", "2", "--workers", str(workers), "--temperature", "0",
        "--sgf-dir", str(tmp_path / "sgf")])
    main()
    files = sorted((tmp_path / "sgf").glob("match-*/*.sgf"))
    assert len(files) == 2
    for path in files:
        nodes, board = replay_sgf(path)
        meta = json.loads(nodes[0]["C"][0])
        assert meta["play_settings"]["move_selection"] == "visit-argmax"
        assert meta["play_settings"]["temperature_override"] == 0
        assert "sha256=" in nodes[0]["PB"][0]
        assert all("chosen is top=True" in n["C"][0] for n in nodes[1:])
        assert board.terminal() or meta["forced_scored"]
    assert '"protocol": "diagnostic-fixed-temperature-v1"' in capsys.readouterr().out

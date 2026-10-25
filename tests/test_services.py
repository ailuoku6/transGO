import json
from transformer_go.config import load_config
from transformer_go.rpc import RPC
from transformer_go.evaluate import wilson_interval
from transformer_go.pipeline import should_promote
from transformer_go.judgment import adjudicate, position_report
from transformer_go._native import Board
import numpy as np
import torch
from transformer_go.model import PolicyValueNet
from transformer_go.search import Evaluator
from transformer_go.play import PlayService, discover_weights
from transformer_go.checkpoint import save_checkpoint


def test_rpc_notifications_ids_and_errors():
    rpc = RPC(load_config("configs/smoke.yaml"), None)
    assert rpc.handle("broken")["error"]["code"] == -32700
    assert rpc.handle("[]")["error"]["code"] == -32600
    def call(method, params=None, **extra):
        return rpc.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}, **extra}))
    assert call("unknown")["error"]["code"] == -32601
    assert call("play")["error"]["code"] == -32602
    assert call("play", {"action": 0})["result"]["player"] == -1
    assert call("play", {"action": 0})["error"]["code"] == -32602
    assert rpc.handle('{"jsonrpc":"2.0","method":"state"}') is None
    assert call("new_game", {"size": 19})["result"]["size"] == 19


def test_promotion_uses_gatekeeper_points_and_accepts_ties():
    result = {"games": 50, "completed": 50, "truncated": 0,
              "wins": 25, "losses": 25, "draws": 0,
              "winrate": 0.5, "confidence95": wilson_interval(25, 50)}
    assert should_promote(result)
    assert not should_promote({**result, "completed": 2})
    assert not should_promote({**result, "truncated": 1})
    assert not should_promote({**result, "wins": 24, "losses": 26})
    # Small engineering smoke matches follow the same acceptance rule.
    assert should_promote({**result, "games": 2, "completed": 2, "wins": 1, "losses": 1})


def test_formal_adjudication_is_rules_only_and_shape_report_is_forecast():
    board = Board(3, 0.5).play(0).play(9).play(9)
    formal = adjudicate(board)
    assert formal["terminal"] and formal["winner"] == 1
    assert formal["score_black"] == board.score()
    own = np.zeros((3, 3, 3), dtype=np.float32)
    own[..., 1] = 1
    own[0, 0, 0] = 0.8; own[0, 0, 1] = 0.1; own[0, 0, 2] = 0.1
    report = position_report(board, {"wdl": np.array([.1, .1, .8]), "score_mean": .5,
                                     "score_logvar": 1., "moves_left": 2., "ownership": own})
    assert report["likely_dead_count"] == 0
    assert np.isclose(report["winrate"], .8)
    with np.testing.assert_raises(ValueError): adjudicate(Board(3))


def test_rpc_analyze_exposes_formal_separate_model_judgment():
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    rpc = RPC(config, Evaluator(PolicyValueNet(config.model), torch.device("cpu")))
    response = rpc.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "analyze",
                                      "params": {"score_utility": 0.05}}))
    report = response["result"]["model"]
    assert set(("wdl", "winrate", "score_mean", "score_std", "ownership",
                "likely_dead", "moves_left")) <= set(report)
    assert len(report["ownership"]) == config.board_size


def test_weight_discovery_defaults_to_newest_best_and_lists_other_checkpoints(tmp_path):
    run = tmp_path / "runs"
    old = run / "old" / "best.pt"
    newer = run / "new" / "best.pt"
    candidate = run / "new" / "candidate-0002.pt"
    for path in (old, newer, candidate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    import os
    os.utime(old, (10, 10))
    os.utime(newer, (20, 20))
    os.utime(candidate, (30, 30))
    weights, default = discover_weights(tmp_path)
    assert default == "runs/new/best.pt"
    assert weights[0]["id"] == default
    assert {item["id"] for item in weights} == {
        "runs/old/best.pt", "runs/new/best.pt", "runs/new/candidate-0002.pt"}


def test_human_game_engine_reply_undo_analysis_and_sgf(tmp_path):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config.search.simulations = 8
    checkpoint = tmp_path / "runs" / "smoke" / "best.pt"
    save_checkpoint(checkpoint, PolicyValueNet(config.model), config)
    alternate_config = load_config("configs/smoke.yaml")
    alternate_config.device = "cpu"
    alternate_config.search.simulations = 4
    alternate = tmp_path / "runs" / "smoke" / "candidate-0001.pt"
    save_checkpoint(alternate, PolicyValueNet(alternate_config.model), alternate_config)
    weights, default = discover_weights(tmp_path)
    game = PlayService(config, torch.device("cpu"), tmp_path, weights, default)

    started = game.request("/api/new", {"weight": default, "size": 3, "komi": 0.5,
                                         "simulations": 8, "human_color": 1})
    assert started["player"] == 1 and started["can_undo"] is False
    response = game.request("/api/play", {"action": 0})
    assert response["player"] == 1 and response["move_number"] == 2
    assert response["can_undo"] is True and response["engine_last"]["simulations"] == 8
    analysis = game.request("/api/analyze", {})
    assert analysis["simulations"] == 8 and analysis["suggestions"]
    json.dumps(analysis)
    undone = game.request("/api/undo", {})
    assert undone["move_number"] == 0 and undone["player"] == 1
    assert undone["can_undo"] is False

    white_start = game.request("/api/new", {"weight": default, "size": 3, "komi": 0.5,
                                             "simulations": 8, "human_color": -1})
    assert white_start["move_number"] == 1 and white_start["player"] == -1
    white_reply = game.request("/api/play", {"action": 8})
    assert white_reply["move_number"] == 3 and white_reply["player"] == -1
    restored = game.request("/api/undo", {})
    assert restored["move_number"] == 1 and restored["player"] == -1
    assert "PB[runs/smoke/best.pt]" in game.sgf()

    alternate_id = "runs/smoke/candidate-0001.pt"
    switched = game.request("/api/new", {"weight": alternate_id, "size": 3,
                                          "komi": 0.5, "human_color": 1})
    assert switched["weight"]["id"] == alternate_id
    assert switched["simulations"] == 4


def test_resignation_is_recorded_as_a_loss_and_cannot_play_afterward(tmp_path):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    checkpoint = tmp_path / "runs" / "smoke" / "best.pt"
    save_checkpoint(checkpoint, PolicyValueNet(config.model), config)
    weights, default = discover_weights(tmp_path)
    game = PlayService(config, torch.device("cpu"), tmp_path, weights, default)
    result = game.request("/api/resign", {})
    assert result["terminal"] and result["result"]["winner"] == -1
    assert "RE[W+R]" in game.sgf()
    with np.testing.assert_raises(ValueError):
        game.request("/api/play", {"action": 0})

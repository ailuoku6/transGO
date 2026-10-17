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

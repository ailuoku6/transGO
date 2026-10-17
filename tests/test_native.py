import numpy as np
import pytest
from transformer_go._native import Board, Search


def test_rules_capture_and_pass_terminal():
    board = Board(3, 0.5)
    assert len(board.legal_mask()) == 10
    assert board.legal(9)
    assert board.play(9).passes == 1
    assert board.play(9).play(9).terminal()


def test_features_and_search_batch():
    board = Board(3, 0.5)
    assert board.features().shape == (21, 3, 3)
    calls = []
    def evaluator(boards):
        calls.append(len(boards))
        return np.ones((len(boards), 10), dtype=np.float32), np.zeros(len(boards), dtype=np.float32)
    result = Search(1.5, 1.0, 2).run(board, evaluator, 8, 4)
    assert result["simulations"] == 8
    assert np.isclose(sum(result["policy"]), 1)
    assert max(calls) > 1


def test_illegal_self_capture():
    board = Board(3, 0.5)
    # A freshly empty point is legal; occupied points are never legal.
    board = board.play(0)
    assert not board.legal(0)


def test_score_utility_is_optional_and_uses_leaf_score():
    board = Board(3, 0.5)

    def evaluator(boards):
        actions = boards[0].size**2+1
        policies = np.ones((len(boards), actions), dtype=np.float32)
        values = np.zeros(len(boards), dtype=np.float32)
        scores = np.ones(len(boards), dtype=np.float32)
        return policies, values, scores

    baseline = Search(seed=3).run(board, evaluator, 16, 4)
    scored = Search(seed=3).run(board, evaluator, 16, 4, score_utility=0.1)
    assert baseline["simulations"] == scored["simulations"] == 16
    assert scored["value"] != baseline["value"]

    with pytest.raises(ValueError, match="score utility"):
        Search().run(board, lambda boards: (np.ones((len(boards), 10), dtype=np.float32),
                                             np.zeros(len(boards), dtype=np.float32)),
                     4, 2, score_utility=0.1)

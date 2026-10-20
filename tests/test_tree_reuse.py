import numpy as np
import pytest

from transformer_go._native import Board, Search
from transformer_go.search import SearchSession, search


class CountingEvaluator:
    def __init__(self, value=0.0):
        self.calls = self.positions = 0
        self.value = value

    def __call__(self, boards):
        self.calls += 1
        self.positions += len(boards)
        return (np.ones((len(boards), boards[0].size**2 + 1), np.float32),
                np.full(len(boards), self.value, np.float32),
                np.zeros(len(boards), np.float32))


def test_promoted_subtree_saves_evaluations_and_counts_edge_visits():
    board = Board(5, .5)
    evaluator = CountingEvaluator()
    tree = Search(seed=3)
    initial = tree.run(board, evaluator, 128, 8, reuse=True)
    action = int(np.argmax(initial["visits"]))
    assert tree.advance(action)
    positions = evaluator.positions
    board = board.play(action)
    result = tree.run(board, evaluator, 128, 8, reuse=True)
    assert result["reused_simulations"] == initial["visits"][action] - 1 > 0
    assert result["simulations"] == sum(result["visits"]) == 128
    assert result["new_simulations"] + result["reused_simulations"] == 128
    assert evaluator.positions - positions == result["new_simulations"]
    fresh = CountingEvaluator()
    Search().run(board, fresh, 128, 8)
    assert evaluator.positions - positions < fresh.positions
    assert sum(result["policy"]) == pytest.approx(1)
    assert result["policy"][action] == 0


def test_repeated_root_and_smaller_budget_do_not_reinfer():
    evaluator, tree, board = CountingEvaluator(), Search(), Board(5)
    first = tree.run(board, evaluator, 64, 8, reuse=True)
    positions = evaluator.positions
    second = tree.run(board, evaluator, 32, 8, reuse=True)
    assert second["visits"] == first["visits"]
    assert second["new_simulations"] == 0 and second["reused_simulations"] == 64
    assert evaluator.positions == positions
    assert sum(second["policy"]) == pytest.approx(1)
    tree.reset()
    assert tree.run(board, evaluator, 32, 8, reuse=True)["reused_simulations"] == 0


@pytest.mark.parametrize("change", ["model", "board", "history", "min_moves", "utility"])
def test_incompatible_state_discards_tree(change):
    root = Board(5)
    board = root.play(0).play(1).play(2).play(3)
    evaluator, tree = CountingEvaluator(), Search()
    tree.run(board, evaluator, 64, 8, reuse=True)
    kwargs = {}
    if change == "model":
        evaluator = CountingEvaluator(.5)
    elif change == "board":
        board = Board.from_position(board.size, board.komi, board.cells, board.to_play)
    elif change == "history":
        other = root.play(2).play(3).play(0).play(1)
        assert other.cells == board.cells and other.hash() == board.hash()
        board = other
    elif change == "min_moves":
        kwargs["min_moves"] = 10
    else:
        kwargs["score_utility"] = .05
    assert tree.run(board, evaluator, 64, 8, reuse=True, **kwargs)["reused_simulations"] == 0


def test_fresh_api_and_missing_child_reset():
    evaluator, tree, board = CountingEvaluator(), Search(), Board(5)
    first = tree.run(board, evaluator, 1, 1, reuse=True)
    missing = next(a for a, n in enumerate(first["visits"]) if n == 0)
    assert not tree.advance(missing)
    assert tree.run(board.play(missing), evaluator, 16, 4, reuse=True)["reused_simulations"] == 0
    assert tree.run(board, evaluator, 16, 4)["new_simulations"] == 16
    assert tree.run(board, evaluator, 16, 4)["new_simulations"] == 16


def test_root_noise_is_not_accumulated_between_calls():
    board, evaluator = Board(5), CountingEvaluator()
    a, b = Search(seed=1), Search(seed=1)
    a.run(board, evaluator, 64, 8, reuse=True)
    b.run(board, evaluator, 64, 8, reuse=True)
    # No new simulations: only alter priors. The next noise draw must replace
    # this root noise, rather than mix new noise into previously noisy priors.
    a.run(board, evaluator, 64, 8, epsilon=.75, reuse=True)
    a.reseed(23)
    b.reseed(23)
    assert a.run(board, evaluator, 128, 8, epsilon=.25, reuse=True) == b.run(
        board, evaluator, 128, 8, epsilon=.25, reuse=True)


def test_failed_evaluator_clears_virtual_reservations():
    base, tree, board = CountingEvaluator(), Search(), Board(5)
    fail = False

    def evaluator(boards):
        if fail:
            raise RuntimeError("test inference failure")
        return base(boards)

    tree.run(board, evaluator, 32, 8, reuse=True)
    fail = True
    with pytest.raises(RuntimeError, match="test inference failure"):
        tree.run(board, evaluator, 64, 8, reuse=True)
    fail = False
    result = tree.run(board, evaluator, 64, 8, reuse=True)
    assert result["new_simulations"] == sum(result["visits"]) == 64
    assert result["reused_simulations"] == 0


def test_evaluation_keeps_separate_trees_and_advances_both(monkeypatch):
    import transformer_go.evaluate as evaluation
    from transformer_go.config import load_config

    sessions = []

    class TrackedSession(SearchSession):
        def __init__(self):
            super().__init__()
            self.moves = []
            sessions.append(self)

        def advance(self, action):
            self.moves.append(action)
            return super().advance(action)

    candidates = [CountingEvaluator(.75), CountingEvaluator(-.75)]
    used = []

    def tracked_search(board, evaluator, config, **kwargs):
        used.append((evaluator, kwargs["session"]))
        return search(board, evaluator, config, **kwargs)

    monkeypatch.setattr(evaluation, "SearchSession", TrackedSession)
    monkeypatch.setattr(evaluation, "search", tracked_search)
    evaluation._play_evaluation_game(load_config("configs/smoke.yaml"), *candidates, 1, 7, 0, 4)
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert sessions[0].moves == sessions[1].moves
    assert len(sessions[0].moves) >= 2
    for evaluator, session in used:
        assert session is sessions[candidates.index(evaluator)]

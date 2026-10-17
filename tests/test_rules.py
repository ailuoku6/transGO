import numpy as np
import pytest
from transformer_go._native import Board, Search


def apply(board, moves):
    for action in moves: board = board.play(action)
    return board


def uniform(boards):
    actions = boards[0].size**2+1
    return np.ones((len(boards), actions), dtype=np.float32), np.zeros(len(boards), dtype=np.float32)


def test_capture_immutable_and_history():
    initial = Board(3, 0.5)
    before = apply(initial, [1, 0])
    after = before.play(3)
    assert after.cells[0] == 0
    assert before.cells[0] == -1 and initial.cells == [0]*9
    assert before.hash() != after.hash()
    assert after.features()[2, 0, 0] == 1


def test_suicide_forbidden():
    board = apply(Board(3, 0.5), [9, 1, 9, 3, 9, 5, 9, 7])
    assert board.to_play == 1 and not board.legal(4)
    with pytest.raises(ValueError, match="illegal"): board.play(4)


def test_positional_superko_and_pass_exemption():
    board = apply(Board(5, 0.5), [1, 6, 5, 2, 11, 12, 25, 8, 7])
    assert board.cells[6] == 0 and board.cells[7] == 1
    assert not board.legal(6) and board.legal(25)
    assert apply(board, [20, 24]).legal(6)


def test_area_ownership_and_terminal():
    empty = Board(3, 0.5)
    assert empty.ownership() == [0]*9 and empty.score() == -0.5
    end = apply(empty, [0, 9, 9])
    assert end.terminal() and not any(end.legal_mask())
    assert end.ownership() == [1]*9
    assert end.score() == 8.5 and end.result() == 1
    with pytest.raises(ValueError): end.play(1)


@pytest.mark.parametrize("budget,batch", [(1, 1), (7, 4), (32, 16), (100, 64)])
def test_exact_visits_batches_and_mask(budget, batch):
    board = Board(3, 0.5).play(0)
    calls = []
    def evaluator(boards):
        calls.append(len(boards)); return uniform(boards)
    result = Search(seed=2).run(board, evaluator, budget, batch)
    assert result["simulations"] == budget and sum(result["visits"]) == budget
    assert result["visits"][0] == 0 and np.isclose(sum(result["policy"]), 1)
    if budget > 1: assert max(calls) > 1


def test_terminal_value_perspective():
    board = Board(3, 0.5).play(9)
    def prefer_pass(boards):
        policies = np.zeros((len(boards), 10), dtype=np.float32); policies[:, 9] = 1
        return policies, np.zeros(len(boards), dtype=np.float32)
    result = Search().run(board, prefer_pass, 20, 4)
    assert result["visits"][9] == 20 and result["value"] == 1


def test_search_validation_and_determinism():
    board = Board(3)
    assert Search(seed=5).run(board, uniform, 32, 8, 0.15, 0.25) == Search(seed=5).run(board, uniform, 32, 8, 0.15, 0.25)
    with pytest.raises(ValueError): Search().run(board, uniform, 0)
    with pytest.raises(ValueError): Search().run(board, lambda b: (np.zeros((1, 2)), np.zeros(1)))
    with pytest.raises(ValueError): Board(-100000)
    with pytest.raises(ValueError): Search(c_puct=float("nan"))
    with pytest.raises(ValueError): Search().run(board, uniform, alpha=float("nan"))


def reference_play(cells, n, player, action, seen):
    """Independent set-based rule oracle."""
    if cells[action]: return None
    cells = list(cells); cells[action] = player
    def neighbors(p):
        r, c = divmod(p, n)
        return {rr*n+cc for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)) if 0 <= rr < n and 0 <= cc < n}
    def group(p):
        stones, liberties, todo = {p}, set(), [p]
        while todo:
            for q in neighbors(todo.pop()):
                if cells[q] == 0: liberties.add(q)
                elif cells[q] == cells[p] and q not in stones: stones.add(q); todo.append(q)
        return stones, liberties
    capture = set()
    for q in neighbors(action):
        if cells[q] == -player:
            stones, liberties = group(q)
            if not liberties: capture.update(stones)
    for p in capture: cells[p] = 0
    if not group(action)[1] or tuple(cells) in seen: return None
    return cells


@pytest.mark.parametrize("n", [3, 5, 9])
def test_random_rules_against_reference(n):
    rng = np.random.default_rng(n)
    for _ in range(12):
        board = Board(n, 0.5); seen = {tuple(board.cells)}
        for _ in range(2*n*n):
            legal = []
            for a in range(n*n):
                result = reference_play(board.cells, n, board.to_play, a, seen)
                assert board.legal(a) == (result is not None)
                if result is not None: legal.append((a, result))
            if not legal: break
            action, expected = legal[int(rng.integers(len(legal)))]
            board = board.play(action)
            assert board.cells == expected
            seen.add(tuple(expected))

import math
from ._native import Board
from .search import search


def wilson_interval(wins, games, z=1.96):
    p = wins/games
    center = (p+z*z/(2*games))/(1+z*z/games)
    radius = z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/(1+z*z/games)
    return max(0, center-radius), min(1, center+radius)


def evaluate(config, candidate, baseline, games=40, report=print, score_utility=0.0):
    """Paired colors and seed-matched randomized openings; no noise/temperature in evaluation."""
    import numpy as np
    if games < 2 or games % 2:
        raise ValueError("evaluation requires an even number of games >= 2")
    wins = losses = draws = truncated = 0
    for g in range(games):
        candidate_color = 1 if g % 2 == 0 else -1
        board = Board(config.board_size, config.komi, config.rules)
        rng = np.random.default_rng(config.seed+g//2)
        # Each pair starts from the same two non-pass legal moves.
        for _ in range(2):
            legal = np.flatnonzero(board.legal_mask()[:-1])
            if len(legal): board = board.play(int(rng.choice(legal)))
        for _ in range(4*board.size*board.size+100):
            evaluator = candidate if board.to_play == candidate_color else baseline
            result = search(board, evaluator, config.search, seed=config.seed+g//2,
                            score_utility=score_utility)
            action = max(range(len(result["visits"])), key=lambda a: result["visits"][a])
            board = board.play(action)
            if board.terminal(): break
        if not board.terminal() or board.no_result(): truncated += 1
        else:
            result = board.result()*candidate_color
            wins += result > 0; losses += result < 0; draws += result == 0
        report({"event": "evaluation_game", "game": g+1, "wins": wins, "losses": losses, "draws": draws, "truncated": truncated})
    completed = wins+losses+draws
    rate = (wins+draws*0.5)/completed if completed else 0.5
    interval = wilson_interval(wins+draws*0.5, completed) if completed else (0.0, 1.0)
    elo = 400*math.log10(rate/(1-rate)) if 0 < rate < 1 else None
    return {"games": games, "completed": completed, "wins": wins, "losses": losses, "draws": draws,
            "truncated": truncated, "winrate": rate, "confidence95": interval, "elo_difference": elo}

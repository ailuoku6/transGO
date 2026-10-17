"""KataGo-style separation of formal adjudication and position judgment.

Formal adjudication is always derived from the native rules engine.  The model
report is a forecast for analysis/search and must never be used as the final
winner in a completed game.
"""
from __future__ import annotations

import numpy as np

from ._native import Board


def adjudicate(board: Board) -> dict:
    """Return the formal result for a completed board.

    The native adapter delegates this to KataGo's BoardHistory.  This function
    deliberately remains rules-only: no network output can change the result.
    """
    if not board.terminal():
        raise ValueError("formal adjudication requires a terminal board")
    formal = board.formal_result()
    score = float(formal["score_black"])
    winner = int(formal["winner"])
    ownership = np.asarray(formal["ownership"], dtype=np.int8).reshape(board.size, board.size)
    return {
        "score_black": score,
        "winner": winner,
        "ownership": ownership.tolist(),
        "dead_stones": np.asarray(formal["dead_stones"], dtype=np.int8).reshape(board.size, board.size).tolist(),
        "territory": np.asarray(formal["territory"], dtype=np.int8).reshape(board.size, board.size).tolist(),
        "seki": np.asarray(formal["seki"], dtype=np.int8).reshape(board.size, board.size).tolist(),
        "rules": str(formal["rules"]),
        "phase": str(formal["phase"]),
        "encore_phase": int(formal["encore_phase"]),
        "adjudication_engine": str(formal["adjudication_engine"]),
        "terminal": True,
        "no_result": bool(formal["no_result"]),
    }


def position_report(board: Board, prediction: dict) -> dict:
    """Convert raw network outputs into a player-relative analysis report.

    Ownership classes are 0=opponent, 1=neutral, 2=current player.  A stone
    whose predicted ownership strongly favors its opponent is reported as a
    *likely* dead stone; it is never removed from formal scoring.
    """
    wdl = np.asarray(prediction["wdl"], dtype=np.float32)
    ownership = np.asarray(prediction["ownership"], dtype=np.float32)
    if wdl.shape != (3,) or ownership.shape != (board.size, board.size, 3):
        raise ValueError("invalid position prediction shapes")
    cells = np.asarray(board.cells, dtype=np.int8).reshape(board.size, board.size)
    confidence = ownership.max(axis=-1)
    dead_probability = np.where(cells == board.to_play, ownership[..., 0], ownership[..., 2])
    likely_dead = (cells != 0) & (dead_probability >= 0.70)
    return {
        "wdl": {"loss": float(wdl[0]), "draw": float(wdl[1]), "win": float(wdl[2])},
        "winrate": float(wdl[2]),
        "value": float(wdl[2] - wdl[0]),
        "score_mean": float(prediction["score_mean"]),
        "score_std": float(prediction["score_logvar"] ** 0.5),
        "moves_left": float(prediction["moves_left"]),
        "ownership": ownership.tolist(),
        "ownership_confidence": confidence.tolist(),
        "dead_probability": dead_probability.tolist(),
        "likely_dead": likely_dead.tolist(),
        "likely_dead_count": int(likely_dead.sum()),
        "player": int(board.to_play),
    }

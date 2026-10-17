"""KataGo gatekeeper match-point acceptance and mathematical early stopping.

Reference: KataGo cpp/command/gatekeeper.cpp, runWriteDataLoop and the
"Candidate wins ties" decision. This is not a sequential significance test.
"""

import math


GATEKEEPER_PROTOCOL = "katago-gatekeeper-v1"
DEFAULT_GATING_GAMES = 200
DEFAULT_REQUIRED_WIN_PROP = 0.5
POINT_EPSILON = 1e-10


def validate_gating_options(games, required_candidate_win_prop=None):
    if not isinstance(games, int) or isinstance(games, bool) or games < 2 or games % 2:
        raise ValueError("evaluation requires an even number of games >= 2")
    if required_candidate_win_prop is not None:
        if (isinstance(required_candidate_win_prop, bool)
                or not math.isfinite(required_candidate_win_prop)
                or not 0 <= required_candidate_win_prop <= 1):
            raise ValueError("required_candidate_win_prop must be finite and in [0, 1]")


def gatekeeper_decision(candidate_points, games_tallied, games_total,
                        required_candidate_win_prop=DEFAULT_REQUIRED_WIN_PROP):
    """Return pending/accepted/rejected using the upstream gatekeeper rules.

An early acceptance requires enough points for the *planned* full match,
even if every remaining game were lost. Rejection requires that winning
every remaining game cannot reach that same target. Ties favor candidate.
"""
    validate_gating_options(games_total, required_candidate_win_prop)
    if (not isinstance(games_tallied, int) or isinstance(games_tallied, bool)
            or not 0 <= games_tallied <= games_total
            or not math.isfinite(candidate_points)
            or not 0 <= candidate_points <= games_tallied):
        raise ValueError("invalid gatekeeper match points or game count")
    if games_tallied == 0:
        return "pending"
    remaining = games_total - games_tallied
    if remaining > 0:
        target = games_total * required_candidate_win_prop
        if candidate_points >= target:
            return "accepted"
        if candidate_points + remaining + POINT_EPSILON < target:
            return "rejected"
        return "pending"
    if candidate_points + POINT_EPSILON < games_tallied * required_candidate_win_prop:
        return "rejected"
    return "accepted"

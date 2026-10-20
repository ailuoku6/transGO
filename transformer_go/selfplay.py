from pathlib import Path
import numpy as np
from ._native import Board
from .search import search, choose_action, SearchSession
from .data import write_shard
from .sgf import dumps


def validate_min_moves(board_size, min_moves, max_moves=None):
    limit = 4*board_size*board_size+100 if max_moves is None else max_moves
    if not isinstance(min_moves, int) or isinstance(min_moves, bool) or min_moves < 0:
        raise ValueError("min_moves must be a nonnegative integer")
    if min_moves > 0 and min_moves+2 > limit:
        raise ValueError(f"min_moves must be <= {limit-2} to leave room for two ending passes")


def play_game(config, evaluator, seed, model_version="untrained", max_moves=None, min_moves=0,
              score_utility=0.0):
    validate_min_moves(config.board_size, min_moves, max_moves)
    rng = np.random.default_rng(seed)
    board = Board(config.board_size, config.komi, config.rules)
    records, actions = [], []
    session = SearchSession()
    max_moves = 4*board.size*board.size+100 if max_moves is None else max_moves
    early_forced_passes = 0
    for t in range(max_moves):
        result = search(board, evaluator, config.search, int(rng.integers(2**31)), noise=True,
                        min_moves=min_moves, score_utility=score_utility, session=session)
        temperature = 1.0 if t < config.search.temperature_moves else 0.1
        action = choose_action(result["policy"], temperature, rng)
        if action == board.size*board.size and t < min_moves:
            early_forced_passes += 1
        records.append((board.features(), board.legal_mask(), result["policy"], board.to_play))
        actions.append(action)
        board = board.play(action)
        session.advance(action)
        if board.terminal():
            break
    if not board.terminal() or board.no_result():
        return None  # Never manufacture final score/ownership from a truncated game.
    n = board.size
    formal = board.formal_result()
    own = np.asarray(formal["ownership"], dtype=np.int8).reshape(n, n)
    winner, score = float(formal["winner"]), float(formal["score_black"])
    data = {
        "features": np.stack([r[0] for r in records]).astype(np.float32),
        "legal": np.asarray([r[1] for r in records], dtype=bool),
        "policy": np.asarray([r[2] for r in records], dtype=np.float32),
        "value": np.asarray([int(winner*r[3])+1 for r in records], dtype=np.int64),
        "score": np.asarray([score*r[3]/(n*n) for r in records], dtype=np.float32),
        "ownership": np.stack([(own*r[3]+1).astype(np.int64) for r in records]),
        "moves_left": np.asarray([(len(records)-i)/(n*n) for i in range(len(records))], dtype=np.float32),
        "aux_mask": np.ones(len(records), dtype=np.float32),
    }
    meta = {"source": "selfplay", "board_size": n, "komi": config.komi, "rules": config.rules,
            "seed": seed, "model_version": model_version, "search": config.to_dict()["search"],
            "score_black": score, "moves": len(records), "min_moves": min_moves,
            "dead_stones": int(np.asarray(formal["dead_stones"], dtype=np.int8).sum()),
            "adjudication_engine": formal["adjudication_engine"],
            "score_utility": score_utility,
            "tree_reuse": True, "search_budget": "retained-plus-new-edge-visits",
            "early_forced_passes": early_forced_passes}
    return data, meta, dumps(n, config.komi, actions, score, config.rules)


def generate(config, evaluator, directory, games=1, seed=None, model_version="untrained", report=print,
             min_moves=0, score_utility=0.0):
    if games < 1:
        raise ValueError("games must be positive")
    validate_min_moves(config.board_size, min_moves)
    completed, attempts = [], 0
    while len(completed) < games and attempts < games*10:
        game_seed = (config.seed if seed is None else seed) + attempts
        attempts += 1
        result = play_game(config, evaluator, game_seed, model_version, min_moves=min_moves,
                           score_utility=score_utility)
        if result is None:
            report({"event": "truncated_game_discarded", "seed": game_seed})
            continue
        data, meta, sgf = result
        path = write_shard(directory, data, meta)
        path.with_suffix(".sgf").write_text(sgf)
        completed.append(path)
        report({"event": "game_completed", "path": str(path), "moves": meta["moves"],
                "score_black": meta["score_black"], "min_moves": min_moves,
                "score_utility": score_utility,
                "early_forced_passes": meta["early_forced_passes"]})
    if len(completed) < games:
        raise RuntimeError("too many truncated games: increase search quality/max moves before training")
    return completed

"""Game state and engine orchestration for the local human-vs-engine UI."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
import time

import numpy as np

from ._native import Board
from .checkpoint import load_checkpoint
from .gtp import to_coord
from .judgment import adjudicate
from .search import Evaluator, SearchSession, analyze_position, choose_action, search
from .sgf import dumps


def discover_weights(root, initial_checkpoint=None):
    root = Path(root).resolve()
    run_root = root / "runs"
    paths = [path.resolve() for path in run_root.rglob("*.pt") if path.is_file()] if run_root.exists() else []
    best = [path for path in paths if path.name == "best.pt"]
    default = max(best, key=lambda path: path.stat().st_mtime_ns) if best else None
    if initial_checkpoint:
        explicit = Path(initial_checkpoint).expanduser().resolve()
        if not explicit.is_file():
            raise ValueError(f"checkpoint not found: {explicit}")
        if explicit not in paths:
            paths.append(explicit)
        default = explicit
    if default is None:
        raise ValueError(f"no best checkpoint found under {run_root}; pass --checkpoint to choose one")
    paths.sort(key=lambda path: (path != default, -path.stat().st_mtime_ns))
    weights = []
    for path in paths:
        try:
            identity = path.relative_to(root).as_posix()
            outside = False
        except ValueError:
            identity = str(path)
            outside = True
        role = ("best" if path.name == "best.pt" else
                "learner" if path.name == "learner.pt" else
                "candidate" if path.name.startswith("candidate-") else "checkpoint")
        weights.append({"id": identity, "label": f"{identity} · {role}", "role": role,
                        "modified": path.stat().st_mtime, "external": outside})
    default_id = next(
        item["id"] for item in weights
        if (Path(item["id"]) if item["external"] else root / item["id"]).resolve() == default
    )
    return weights, default_id


class PlayService:
    def __init__(self, config, device, root, weights, default_id):
        self.root = Path(root).resolve()
        self.device = device
        self.weights = list(weights)
        self.weight_paths = {}
        for item in self.weights:
            identity = item["id"]
            path = Path(identity) if item["external"] else self.root / identity
            self.weight_paths[identity] = path.resolve()
        self.lock = threading.RLock()
        self.base_config = config
        self.rng = np.random.default_rng(config.seed)
        self.weight_id = None
        self.model = None
        self.evaluator = None
        self.config = None
        self.search_config = None
        self.checkpoint_info = None
        self.board = None
        self.history = []
        self.actions = []
        self.move_info = []
        self.human_color = 1
        self.session = SearchSession()
        self.resigned = False
        self.resigned_color = None
        self._load_weight(default_id)
        self._reset_game(self.config.board_size, self.config.komi, self.config.search.simulations, 1)

    def _load_weight(self, identity):
        if identity not in self.weight_paths:
            raise ValueError("unknown checkpoint")
        model, config, payload = load_checkpoint(self.weight_paths[identity], self.device)
        self.weight_id = identity
        self.model, self.config = model, config
        self.evaluator = Evaluator(model, self.device)
        self.search_config = deepcopy(config.search)
        self.checkpoint_info = {"id": identity, "step": int(payload.get("step", 0)),
                                "board_size": config.board_size, "komi": config.komi,
                                "rules": config.rules, "default_simulations": config.search.simulations}

    def _reset_game(self, size, komi, simulations, human_color):
        if isinstance(size, bool) or not isinstance(size, int) or not 2 <= size <= 19:
            raise ValueError("board size must be an integer in [2,19]")
        if isinstance(simulations, bool) or not isinstance(simulations, int) or not 1 <= simulations <= 5000:
            raise ValueError("simulations must be an integer in [1,5000]")
        if isinstance(human_color, bool) or human_color not in (-1, 1):
            raise ValueError("human_color must be 1 (black) or -1 (white)")
        komi = float(komi)
        if not np.isfinite(komi) or abs(komi) > 100:
            raise ValueError("komi must be finite and between -100 and 100")
        self.search_config = deepcopy(self.config.search)
        self.search_config.simulations = simulations
        self.board = Board(size, komi, self.config.rules)
        self.history = [self.board]
        self.actions = []
        self.move_info = []
        self.human_color = human_color
        self.session = SearchSession()
        self.resigned = False
        self.resigned_color = None

    def _apply(self, action, actor):
        if not self.board.legal(action):
            raise ValueError("illegal move")
        self.actions.append(action)
        self.move_info.append({"actor": actor, "action": action, "visits": None, "simulations": None,
                               "seconds": None})
        self.board = self.board.play(action)
        self.history.append(self.board)
        self.session.advance(action)

    def _engine_move(self):
        started = time.perf_counter()
        result = search(self.board, self.evaluator, self.search_config,
                        seed=int(self.rng.integers(2**31)), score_utility=0.05, session=self.session)
        action = choose_action(result["policy"], 0, self.rng)
        self._apply(action, "engine")
        details = {"action": action, "coordinate": to_coord(action, self.board.size),
                   "visits": int(result["visits"][action]), "simulations": int(result["simulations"]),
                   "seconds": time.perf_counter() - started,
                   "root_value": float(result["value"])}
        self.move_info[-1].update({key: details[key] for key in ("visits", "simulations", "seconds")})
        return details

    def new_game(self, payload):
        identity = payload.get("weight", self.weight_id)
        if identity != self.weight_id:
            self._load_weight(identity)
        self._reset_game(payload.get("size", self.config.board_size),
                         payload.get("komi", self.config.komi),
                         payload.get("simulations", self.config.search.simulations),
                         payload.get("human_color", 1))
        engine_move = None
        if self.board.to_play != self.human_color:
            engine_move = self._engine_move()
        state = self.state()
        state["engine_move"] = engine_move
        return state

    def play(self, payload):
        if self.resigned or self.board.terminal():
            raise ValueError("game is over")
        if self.board.to_play != self.human_color:
            raise ValueError("it is not the human player's turn")
        action = payload.get("action")
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action <= self.board.size * self.board.size:
            raise ValueError("action must be a board point or pass")
        self._apply(action, "human")
        engine_move = None
        if not self.board.terminal():
            engine_move = self._engine_move()
        state = self.state()
        state["engine_move"] = engine_move
        return state

    def undo(self):
        human_moves = [index for index, item in enumerate(self.move_info) if item["actor"] == "human"]
        if not human_moves:
            raise ValueError("no moves to undo")
        count = len(self.actions) - human_moves[-1]
        del self.actions[-count:]
        del self.move_info[-count:]
        del self.history[-count:]
        self.board = self.history[-1]
        self.session.reset()
        self.session = SearchSession()
        self.resigned = False
        self.resigned_color = None
        return self.state()

    def resign(self):
        if self.resigned or self.board.terminal():
            raise ValueError("game is over")
        self.resigned = True
        self.resigned_color = self.human_color
        return self.state()

    def analyze(self):
        if self.resigned or self.board.terminal():
            raise ValueError("cannot analyze a finished game")
        if self.board.to_play != self.human_color:
            raise ValueError("analysis is available on the human turn")
        result = analyze_position(self.board, self.evaluator, self.search_config,
                                  seed=int(self.rng.integers(2**31)), score_utility=0.05)
        visits = np.asarray(result["visits"], dtype=np.int64)
        policy = np.asarray(result["policy"], dtype=np.float64)
        ranked = np.argsort(visits)[::-1][:5]
        return {"suggestions": [{"action": int(action), "coordinate": to_coord(int(action), self.board.size),
                                 "visits": int(visits[action]), "policy": float(policy[action])}
                                for action in ranked if visits[action] > 0],
                "model": {"winrate": float(result["model"]["winrate"]),
                          "score_mean": float(result["model"]["score_mean"]),
                          "score_std": float(result["model"]["score_std"]),
                          "moves_left": float(result["model"]["moves_left"])},
                "simulations": int(result["simulations"])}

    def state(self):
        terminal = self.board.terminal() or self.resigned
        payload = {"weight": self.checkpoint_info, "size": self.board.size, "komi": self.board.komi,
                   "rules": self.board.rules, "cells": list(self.board.cells),
                   "legal": list(self.board.legal_mask()), "player": self.board.to_play,
                   "human_color": self.human_color, "move_number": self.board.move_number,
                   "phase": self.board.phase(), "encore_phase": self.board.encore_phase,
                   "terminal": terminal, "resigned": self.resigned,
                   "last_action": self.actions[-1] if self.actions else None,
                   "can_undo": any(item["actor"] == "human" for item in self.move_info),
                   "actions": list(self.actions), "simulations": self.search_config.simulations,
                   "engine_last": next(({
                       "action": item["action"], "coordinate": to_coord(item["action"], self.board.size),
                       "visits": item["visits"], "simulations": item["simulations"],
                       "seconds": item["seconds"]}
                       for item in reversed(self.move_info) if item["actor"] == "engine"), None)}
        if terminal:
            if self.resigned:
                payload["result"] = {"winner": -self.resigned_color, "reason": "resignation"}
            else:
                report = adjudicate(self.board)
                payload["adjudication"] = report
                payload["result"] = {"winner": report["winner"], "score_black": report["score_black"],
                                     "no_result": report["no_result"], "reason": "two_passes"}
        return payload

    def sgf(self):
        if self.resigned:
            winner = -self.resigned_color
            result = ("B" if winner == 1 else "W") + "+R"
        elif self.board.terminal():
            if self.board.no_result():
                result = "Void"
            else:
                score = self.board.score()
                result = "0" if score == 0 else f"{'B' if score > 0 else 'W'}+{abs(score):g}"
        else:
            result = ""
        comments = []
        for item in self.move_info:
            if item["actor"] == "engine":
                comments.append(f"MCTS visits={item['visits']}/{item['simulations']} time={item['seconds']:.3f}s")
            else:
                comments.append("Human move")
        return dumps(self.board.size, self.board.komi, self.actions, rules=self.board.rules,
                     result=result, black_name="Human" if self.human_color == 1 else self.weight_id,
                     white_name="Human" if self.human_color == -1 else self.weight_id,
                     comment=f"checkpoint={self.weight_id}; step={self.checkpoint_info['step']}; simulations={self.search_config.simulations}",
                     move_comments=comments)

    def request(self, path, payload):
        if path == "/api/new":
            return self.new_game(payload)
        if path == "/api/play":
            return self.play(payload)
        if path == "/api/undo":
            return self.undo()
        if path == "/api/resign":
            return self.resign()
        if path == "/api/analyze":
            return self.analyze()
        raise ValueError("unknown API endpoint")

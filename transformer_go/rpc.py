"""Line-delimited JSON-RPC 2.0 for a local GUI or process client."""
import json
import sys
from ._native import Board
from .search import analyze_position
from .judgment import adjudicate


class RPC:
    def __init__(self, config, evaluator):
        self.config, self.evaluator = config, evaluator
        self.board = Board(config.board_size, config.komi, config.rules)

    def dispatch(self, method, params):
        if method == "new_game":
            self.board = Board(params.get("size", self.config.board_size), params.get("komi", self.config.komi),
                               params.get("rules", self.config.rules))
        elif method == "play": self.board = self.board.play(int(params["action"]))
        elif method == "analyze":
            score_utility = float(params.get("score_utility", 0.05))
            result = analyze_position(self.board, self.evaluator, self.config.search,
                                      self.config.seed, score_utility)
            result["player"] = self.board.to_play
            return result
        elif method != "state": raise NotImplementedError("method not found")
        result = {"size": self.board.size, "komi": self.board.komi, "rules": self.board.rules,
                  "phase": self.board.phase(), "encore_phase": self.board.encore_phase,
                  "player": self.board.to_play, "cells": self.board.cells,
                  "terminal": self.board.terminal(), "legal": self.board.legal_mask()}
        if self.board.terminal():
            result["adjudication"] = adjudicate(self.board)
        return result

    def handle(self, line):
        identifier, notification = None, False
        try:
            try: request = json.loads(line)
            except json.JSONDecodeError: return self.error(None, -32700, "parse error")
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
                return self.error(None, -32600, "invalid request")
            identifier = request.get("id")
            notification = "id" not in request
            params = request.get("params", {})
            if not isinstance(params, dict): raise ValueError("params must be an object")
            supported = {"new_game": {"size", "komi", "rules"}, "play": {"action"}, "state": set(),
                         "analyze": {"score_utility"}}
            if request["method"] in supported and set(params)-supported[request["method"]]:
                raise ValueError("unknown parameter")
            result = self.dispatch(request["method"], params)
            return None if notification else {"jsonrpc": "2.0", "id": identifier, "result": result}
        except NotImplementedError as exc:
            return None if notification else self.error(identifier, -32601, str(exc))
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return None if notification else self.error(identifier, -32602, str(exc))

    @staticmethod
    def error(identifier, code, message):
        return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}

    def run(self, source=sys.stdin, target=sys.stdout):
        for line in source:
            response = self.handle(line)
            if response is not None:
                target.write(json.dumps(response)+"\n"); target.flush()

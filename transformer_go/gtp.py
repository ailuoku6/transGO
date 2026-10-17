import re
import sys
from ._native import Board
from .search import search, choose_action
import numpy as np

LETTERS = "ABCDEFGHJKLMNOPQRST"


def parse_color(color):
    color = color.lower()
    if color in ("b", "black"): return 1
    if color in ("w", "white"): return -1
    raise ValueError("expected black or white")


def to_action(coord, size):
    coord = coord.upper()
    if coord == "PASS": return size*size
    if not re.fullmatch(r"[A-HJ-T][1-9][0-9]?", coord):
        raise ValueError("invalid GTP coordinate")
    col, row = LETTERS.index(coord[0]), size-int(coord[1:])
    if not (0 <= row < size and col < size): raise ValueError("coordinate outside board")
    return row*size+col


def to_coord(action, size):
    if action == size*size: return "pass"
    row, col = divmod(action, size)
    return f"{LETTERS[col]}{size-row}"


class GTP:
    commands = ("protocol_version", "name", "version", "known_command", "list_commands", "boardsize",
                "clear_board", "komi", "play", "genmove", "showboard", "final_score",
                "final_status_list", "undo", "quit")

    def __init__(self, config, evaluator):
        self.config, self.evaluator = config, evaluator
        self.board = Board(config.board_size, config.komi, config.rules)
        self.undo = []
        self.rng = np.random.default_rng(config.seed)

    def execute(self, command, args):
        command = command.lower()
        if command not in self.commands: raise ValueError("unknown command")
        arity = {"known_command": 1, "boardsize": 1, "komi": 1, "play": 2, "genmove": 1,
                 "final_status_list": 1}
        if len(args) != arity.get(command, 0): raise ValueError("wrong number of arguments")
        if command == "protocol_version": return "2"
        if command == "name": return "transformerGo"
        if command == "version": return "0.1.0"
        if command == "known_command": return "true" if args[0] in self.commands else "false"
        if command == "list_commands": return "\n".join(self.commands)
        if command == "boardsize":
            n = int(args[0]); self.board = Board(n, self.board.komi, self.config.rules); self.undo.clear(); return ""
        if command == "clear_board":
            self.board = Board(self.board.size, self.board.komi, self.config.rules); self.undo.clear(); return ""
        if command == "komi":
            if self.board.move_number: raise ValueError("set komi before playing; clear_board first")
            self.board = Board(self.board.size, float(args[0]), self.config.rules); return ""
        if command in ("play", "genmove"):
            if parse_color(args[0]) != self.board.to_play: raise ValueError("out of turn")
            if command == "play": action = to_action(args[1], self.board.size)
            else:
                result = search(self.board, self.evaluator, self.config.search, self.config.seed)
                action = choose_action(result["policy"], 0, self.rng)
            new_board = self.board.play(action)
            self.undo.append(self.board); self.board = new_board
            return to_coord(action, self.board.size) if command == "genmove" else ""
        if command == "undo":
            if not self.undo: raise ValueError("cannot undo")
            self.board = self.undo.pop(); return ""
        if command == "showboard":
            symbols = {0: ".", 1: "X", -1: "O"}
            lines = [f"{self.board.size-r:2} " + " ".join(symbols[c] for c in self.board.cells[r*self.board.size:(r+1)*self.board.size]) for r in range(self.board.size)]
            return "\n" + "\n".join(lines) + "\n   " + " ".join(LETTERS[:self.board.size])
        if command == "final_score":
            if not self.board.terminal(): raise ValueError("game has not ended with two passes")
            if self.board.no_result(): return "Void"
            score = self.board.score()
            return "0" if score == 0 else f"{'B' if score>0 else 'W'}+{abs(score):g}"
        if command == "final_status_list":
            if args[0].lower() not in ("alive", "dead", "seki"):
                raise ValueError("argument must be alive, dead, or seki")
            dead = self.board.dead_stones()
            seki = self.board.seki()
            points = []
            for action, color in enumerate(self.board.cells):
                if args[0].lower() == "dead" and color != 0 and dead[action]:
                    points.append(to_coord(action, self.board.size))
                elif args[0].lower() == "alive" and color != 0 and not dead[action]:
                    points.append(to_coord(action, self.board.size))
                elif args[0].lower() == "seki" and color != 0 and seki[action]:
                    points.append(to_coord(action, self.board.size))
            return " ".join(points)
        return ""

    def run(self, source=sys.stdin, target=sys.stdout):
        for line in source:
            words = line.split("#", 1)[0].split()
            if not words: continue
            identifier = words.pop(0) if words[0].isdigit() else ""
            try:
                if not words: raise ValueError("missing command")
                command = words.pop(0).lower()
                answer = self.execute(command, words)
                target.write(f"={identifier} {answer}\n\n"); target.flush()
                if command == "quit": return
            except (ValueError, RuntimeError, IndexError) as exc:
                target.write(f"?{identifier} {exc}\n\n"); target.flush()

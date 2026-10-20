"""Export and inspect game records; SGF is not a training input."""
import re


def _escape(value):
    return str(value).replace("\\", "\\\\").replace("]", "\\]")


def dumps(size, komi, actions, score=None, rules="Chinese", *, result=None,
          black_name=None, white_name=None, comment=None, move_comments=None):
    if result is None:
        result = "" if score is None else ("0" if score == 0 else f"{'B' if score>0 else 'W'}+{abs(score):g}")
    root = f"(;GM[1]FF[4]CA[UTF-8]SZ[{size}]KM[{komi:g}]RU[{_escape(rules)}]RE[{_escape(result)}]"
    for name, value in (("PB", black_name), ("PW", white_name), ("C", comment)):
        if value is not None:
            root += f"{name}[{_escape(value)}]"
    if move_comments is not None and len(move_comments) != len(actions):
        raise ValueError("move comments must match the number of actions")
    for i, action in enumerate(actions):
        point = "" if action == size*size else chr(97+action % size) + chr(97+action // size)
        root += f";{'B' if i % 2 == 0 else 'W'}[{point}]"
        if move_comments is not None:
            root += f"C[{_escape(move_comments[i])}]"
    return root + ")\n"


def parse(text):
    pos, nodes, current, depth = 0, [], None, 0
    while pos < len(text):
        c = text[pos]
        if c.isspace():
            pos += 1
            continue
        if c == "(":
            depth += 1
            if depth != 1 or nodes:
                raise ValueError("SGF variations/collections unsupported; export a single mainline")
            pos += 1
        elif c == ")":
            depth -= 1
            if depth != 0:
                raise ValueError("invalid SGF parentheses")
            pos += 1
        elif c == ";":
            if depth != 1:
                raise ValueError("node outside SGF tree")
            current = {}
            nodes.append(current)
            pos += 1
        elif c.isupper() and current is not None:
            match = re.match(r"[A-Z]+", text[pos:])
            name = match.group()
            if name in current:
                raise ValueError("duplicate SGF property")
            pos += len(name)
            values = []
            while pos < len(text):
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos >= len(text) or text[pos] != "[":
                    break
                pos += 1
                value = ""
                while pos < len(text) and text[pos] != "]":
                    if text[pos] == "\\":
                        pos += 1
                        if pos >= len(text):
                            raise ValueError("unterminated SGF escape")
                    value += text[pos]
                    pos += 1
                if pos >= len(text):
                    raise ValueError("unterminated SGF property")
                values.append(value)
                pos += 1
            if not values:
                raise ValueError("missing SGF property value")
            current[name] = values
        else:
            raise ValueError(f"invalid SGF character at {pos}")
    if depth != 0 or not nodes:
        raise ValueError("incomplete SGF")
    return nodes

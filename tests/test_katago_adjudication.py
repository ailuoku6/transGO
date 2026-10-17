import io
import json

import numpy as np
from transformer_go._native import Board
from transformer_go.config import load_config
from transformer_go.data import ADJUDICATION_VERSION, load_replay, read_shard, write_shard
from transformer_go.gtp import GTP
from transformer_go.sgf import dumps


def test_katago_area_fixture_matches_native_board_area():
    rows = [
        "..o.o.xx.",
        ".oooo.x.x",
        "oo.....xx",
        ".........",
        ".........",
        "x......oo",
        ".xx...o.o",
        "x.x..oo.o",
        ".x.x.o.o.",
    ]
    cells = [{"x": 1, "o": -1, ".": 0}[c] for row in rows for c in row]
    board = Board.from_position(9, 0.0, cells, 1, "chinese-area-psk")
    expected = [
        "OOOOO.XXX",
        "OOOOO.XXX",
        "OO.....XX",
        ".........",
        ".........",
        "X......OO",
        "XXX...OOO",
        "XXX..OOOO",
        "XXXX.OOOO",
    ]
    actual = board.ownership()
    assert [
        "".join("X" if c == 1 else "O" if c == -1 else "." for c in actual[y * 9:(y + 1) * 9])
        for y in range(9)
    ] == expected


def test_katago_area_marks_dead_throw_in():
    # This is the 7x7 throw-in position used by KataGo's
    # testexcludeterritoryatari.cpp.  The black stone at A5 is pass-dead in
    # the white pass-alive area under Chinese area scoring.
    rows = [
        ".oo.ox.",
        "oo.oxx.",
        "xoooox.",
        ".oooox.",
        "oooo.x.",
        "o.ooox.",
        "ooooox.",
    ]
    cells = [{"x": 1, "o": -1, ".": 0}[c] for row in rows for c in row]
    board = Board.from_position(7, 0.0, cells, 1, "chinese-area-psk")
    dead = board.dead_stones()
    assert dead[2 * 7] == 1  # A5
    formal = board.formal_result()
    assert formal["adjudication_engine"] == "katago-boardhistory"
    assert formal["ownership"][2 * 7] == -1


def test_territory_encore_uses_three_katago_phases():
    board = Board(3, 0.5, "japanese")
    assert board.phase() == "normal_play"
    board = board.play(9).play(9)
    assert board.phase() == "encore1" and not board.terminal()
    board = board.play(9).play(9)
    assert board.phase() == "encore2" and not board.terminal()
    board = board.play(9).play(9)
    assert board.phase() == "finished" and board.terminal()


def test_gtp_exposes_formal_dead_status():
    config = load_config("configs/smoke.yaml")
    engine = GTP(config, None)
    rows = [
        ".oo.ox.",
        "oo.oxx.",
        "xoooox.",
        ".oooox.",
        "oooo.x.",
        "o.ooox.",
        "ooooox.",
    ]
    cells = [{"x": 1, "o": -1, ".": 0}[c] for row in rows for c in row]
    engine.board = Board.from_position(7, 0.0, cells, 1, "chinese-area-psk")
    assert engine.execute("final_status_list", ["dead"]) == "A5"
    output = io.StringIO()
    engine.run(io.StringIO("1 known_command final_status_list\n2 quit\n"), output)
    assert "=1 true" in output.getvalue()


def test_legacy_replay_is_relabelled_in_cache_without_touching_source(tmp_path):
    n = 3
    features = np.zeros((2, 21, n, n), dtype=np.float32)
    features[0, 18] = 1  # black to play
    legal = np.ones((2, n * n + 1), dtype=bool)
    policy = np.zeros((2, n * n + 1), dtype=np.float32)
    policy[:, -1] = 1
    data = {
        "features": features,
        "legal": legal,
        "policy": policy,
        "value": np.array([2, 0], dtype=np.int64),
        "score": np.array([0.1, -0.1], dtype=np.float32),
        "ownership": np.ones((2, n, n), dtype=np.int64),
        "moves_left": np.ones(2, dtype=np.float32),
        "aux_mask": np.ones(2, dtype=np.float32),
    }
    path = write_shard(tmp_path, data, {
        "source": "selfplay", "board_size": n, "komi": 0.5,
        "rules": "chinese-area-psk", "model_version": "old",
    })
    path.with_suffix(".sgf").write_text(dumps(n, 0.5, [n * n, n * n], -0.5))
    with np.load(path, allow_pickle=False) as source:
        old_data = {key: source[key] for key in (
            "features", "legal", "policy", "value", "score", "ownership",
            "moves_left", "aux_mask",
        )}
        old_meta = json.loads(str(source["metadata"]))
    old_meta.pop("adjudication_version")
    np.savez_compressed(path, **old_data, metadata=np.asarray(json.dumps(old_meta)))

    replay = load_replay(tmp_path, n, 100)
    assert replay["value"].tolist() == [0, 2]
    assert (tmp_path / ".adjudication-cache" / path.name).exists()
    # Migration is additive: the original shard remains present and untouched
    # in meaning (its deliberately old labels are still there).
    with np.load(path, allow_pickle=False) as source:
        assert source["value"].tolist() == [2, 0]
    _, cached_meta = read_shard(tmp_path / ".adjudication-cache" / path.name)
    assert cached_meta["adjudication_version"] == ADJUDICATION_VERSION

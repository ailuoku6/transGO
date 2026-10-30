import json
import os

import numpy as np
import pytest

from scripts.dedupe_replay import delete_duplicates, scan_duplicates
from transformer_go._native import Board
from transformer_go.data import write_shard


def _data(value=0):
    board = Board(3, 0.5)
    return {"features": board.features()[None],
            "legal": np.asarray(board.legal_mask())[None],
            "policy": np.full((1, 10), 0.1, dtype=np.float32),
            "value": np.array([value]), "score": np.zeros(1, dtype=np.float32),
            "ownership": np.ones((1, 3, 3), dtype=np.int64),
            "moves_left": np.ones(1, dtype=np.float32),
            "aux_mask": np.ones(1, dtype=np.float32)}


def _write(root, value, mtime):
    path = write_shard(root, _data(value), {"source": "selfplay", "board_size": 3,
                                           "seed": int(mtime)})
    path.with_suffix(".sgf").write_text("(;SZ[3])")
    os.utime(path, ns=(mtime, mtime))
    return path


def test_dedupe_keeps_newest_exact_copy_and_removes_associated_files(tmp_path):
    replay = tmp_path / "replay"
    old = _write(replay, 0, 1_000_000_000)
    new = _write(replay, 0, 2_000_000_000)
    distinct = _write(replay, 2, 3_000_000_000)
    cache = replay / ".adjudication-cache" / old.name
    cache.parent.mkdir()
    cache.write_bytes(b"cached")

    files, duplicates = scan_duplicates(replay)
    assert len(files) == 3
    assert [(file.path, keeper.path) for file, keeper, *_ in duplicates] == [(old, new)]
    assert old.exists()  # Scanning is read-only.
    manifest = tmp_path / "analysis" / "manifest.jsonl"
    result = delete_duplicates(replay, files, duplicates, manifest)
    assert result["removed"] == 1
    assert not old.exists() and not old.with_suffix(".sgf").exists() and not cache.exists()
    assert new.exists() and new.with_suffix(".sgf").exists() and distinct.exists()
    entry = json.loads(manifest.read_text().strip())
    assert entry["discarded"] == old.name and entry["kept"] == new.name


def test_dedupe_does_not_discard_distinct_arrays_on_signature_collision(tmp_path, monkeypatch):
    replay = tmp_path / "replay"
    old = _write(replay, 0, 1_000_000_000)
    new = _write(replay, 2, 2_000_000_000)
    monkeypatch.setattr("scripts.dedupe_replay._training_signature", lambda _: (0,))
    files, duplicates = scan_duplicates(replay)
    assert len(files) == 2 and duplicates == []
    result = delete_duplicates(replay, files, duplicates, tmp_path / "manifest.jsonl")
    assert result["removed"] == 0 and old.exists() and new.exists()


def test_dedupe_refuses_changed_file_before_deletion(tmp_path):
    replay = tmp_path / "replay"
    old = _write(replay, 0, 1_000_000_000)
    _write(replay, 0, 2_000_000_000)
    files, duplicates = scan_duplicates(replay)
    old.write_bytes(old.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="changed since scan"):
        delete_duplicates(replay, files, duplicates, tmp_path / "manifest.jsonl")
    assert old.exists()


def test_dedupe_refuses_replay_directory_symlink(tmp_path):
    replay = tmp_path / "replay"
    replay.mkdir()
    linked = tmp_path / "linked-replay"
    linked.symlink_to(replay, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        scan_duplicates(linked)


def test_dedupe_refuses_associated_symlink_before_any_deletion(tmp_path):
    replay = tmp_path / "replay"
    old = _write(replay, 0, 1_000_000_000)
    new = _write(replay, 0, 2_000_000_000)
    old.with_suffix(".sgf").unlink()
    old.with_suffix(".sgf").symlink_to(new.with_suffix(".sgf"))
    files, duplicates = scan_duplicates(replay)
    with pytest.raises(ValueError, match="associated symlink"):
        delete_duplicates(replay, files, duplicates, tmp_path / "manifest.jsonl")
    assert old.exists() and new.exists()
    assert not (tmp_path / "manifest.jsonl").exists()

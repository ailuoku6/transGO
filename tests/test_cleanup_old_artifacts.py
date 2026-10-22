from datetime import datetime, timezone
import os
import time

import pytest

from scripts.cleanup_old_artifacts import cleanup, main


NOW = datetime(2026, 10, 22, 12, tzinfo=timezone.utc).timestamp()


def touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    os.utime(path, (mtime, mtime))
    return path


def test_cleanup_preview_and_delete_are_scoped_and_keep_current_checkpoints(tmp_path):
    root = tmp_path / "run"
    old = NOW - 25 * 60 * 60
    recent = NOW - 23 * 60 * 60
    old_candidate = touch(root / "candidate-0001.pt", old)
    old_best = touch(root / "best.pt", old)
    old_learner = touch(root / "learner.pt", old)
    recent_candidate = touch(root / "candidate-0002.pt", recent)
    old_npz = touch(root / "replay" / "old.npz", old)
    recent_sgf = touch(root / "replay" / "recent.sgf", recent)
    old_sgf = touch(root / "replay" / "old.sgf", old)
    unrelated = touch(root / "evaluation" / "old.sgf", old)
    old_log = touch(root / "training" / "old.jsonl", old)
    nested_checkpoint = touch(root / "inspection" / "snapshot.pt", old)

    preview = cleanup(root, now=NOW)
    assert {path for path, _, _ in preview["candidates"]} == {old_candidate, old_npz, old_sgf}
    assert all(path.exists() for path, _, _ in preview["candidates"])

    deleted = cleanup(root, delete=True, now=NOW)
    assert {path for path, _, _ in deleted["removed"]} == {old_candidate, old_npz, old_sgf}
    assert not old_candidate.exists() and not old_npz.exists() and not old_sgf.exists()
    assert old_best.exists() and old_learner.exists() and recent_candidate.exists()
    assert recent_sgf.exists() and unrelated.exists() and old_log.exists()
    assert nested_checkpoint.exists()


@pytest.mark.parametrize("age", [24 * 60 * 60, 23 * 60 * 60])
def test_cleanup_keeps_files_at_or_newer_than_cutoff(tmp_path, age):
    path = touch(tmp_path / "run" / "candidate.pt", NOW - age)
    result = cleanup(path.parent, now=NOW)
    assert result["candidates"] == []
    assert path.exists()


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf")])
def test_cleanup_rejects_invalid_age(tmp_path, hours):
    with pytest.raises(ValueError, match="older_than_hours"):
        cleanup(tmp_path, older_than_hours=hours, now=NOW)


def test_cli_defaults_to_preview(tmp_path, capsys):
    root = tmp_path / "run"
    old = touch(root / "candidate.pt", time.time() - 25 * 60 * 60)
    assert main(["--root", str(root)]) == 0
    assert "Would remove 1 files" in capsys.readouterr().out
    assert old.exists()
    assert main(["--root", str(root), "--delete"]) == 0
    assert "Removed 1 files" in capsys.readouterr().out
    assert not old.exists()


def test_cleanup_never_follows_replay_or_file_symlinks(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    external = touch(tmp_path / "external" / "old.npz", NOW - 25 * 60 * 60)
    (root / "replay").symlink_to(external.parent, target_is_directory=True)
    assert cleanup(root, delete=True, now=NOW)["removed"] == []
    (root / "replay").unlink()
    (root / "replay").mkdir()
    (root / "replay" / "external").symlink_to(external.parent, target_is_directory=True)
    (root / "replay" / "linked.npz").symlink_to(external)
    assert cleanup(root, delete=True, now=NOW)["removed"] == []
    assert external.exists()


def test_cleanup_skips_file_updated_after_scan(tmp_path, monkeypatch):
    import scripts.cleanup_old_artifacts as module

    path = touch(tmp_path / "run" / "candidate.pt", NOW - 25 * 60 * 60)
    original_find = module.find_old_artifacts

    def scan_then_update(root, cutoff):
        candidates = original_find(root, cutoff)
        os.utime(path, (NOW, NOW))
        return candidates

    monkeypatch.setattr(module, "find_old_artifacts", scan_then_update)
    assert cleanup(path.parent, delete=True, now=NOW)["removed"] == []
    assert path.exists()

#!/usr/bin/env python3
"""Preview or remove old model checkpoints and self-play replay artifacts."""

import argparse
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runs" / "zero-9x9"
PROTECTED_CHECKPOINTS = {"best.pt", "learner.pt"}


def find_old_artifacts(root: Path, cutoff: float):
    """Return old target files as (path, size, mtime), without following symlinks."""
    replay = root / "replay"
    candidates = []
    scopes = [list(root.glob("*.pt"))]
    if replay.is_dir() and not replay.is_symlink():
        # os.walk does not descend into directory symlinks by default.
        scopes.append([Path(directory) / name
                       for directory, _, names in os.walk(replay, followlinks=False)
                       for name in names if Path(name).suffix in {".npz", ".sgf"}])
    for paths in scopes:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                continue
            if path.parent == root and path.name in PROTECTED_CHECKPOINTS:
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if stat.st_mtime < cutoff:
                candidates.append((path, stat.st_size, stat.st_mtime))
    return sorted(candidates, key=lambda item: (item[2], str(item[0])))


def cleanup(root: Path, older_than_hours=24, delete=False, now=None):
    """Preview old files by default; unlink them only when delete=True."""
    if not math.isfinite(older_than_hours) or older_than_hours <= 0:
        raise ValueError("older_than_hours must be a positive finite number")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    cutoff = now - timedelta(hours=older_than_hours).total_seconds()
    candidates = find_old_artifacts(root, cutoff)

    removed = []
    errors = []
    for path, size, mtime in candidates:
        if not delete:
            continue
        try:
            # Recheck age immediately before unlinking in case a writer updated it.
            if path.is_symlink() or path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed.append((path, size, mtime))
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append((path, str(exc)))

    return {
        "root": root,
        "cutoff": datetime.fromtimestamp(cutoff, timezone.utc),
        "older_than_hours": older_than_hours,
        "delete": delete,
        "candidates": candidates,
        "removed": removed,
        "errors": errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"run directory (default: {DEFAULT_ROOT})")
    parser.add_argument("--older-than-hours", type=float, default=24,
                        help="remove files older than this many hours (default: 24)")
    parser.add_argument("--delete", action="store_true",
                        help="perform deletion; without this flag only preview files")
    args = parser.parse_args(argv)

    try:
        result = cleanup(args.root, args.older_than_hours, args.delete)
    except ValueError as exc:
        parser.error(str(exc))

    action = "Removed" if args.delete else "Would remove"
    files = result["removed"] if args.delete else result["candidates"]
    total_bytes = sum(size for _, size, _ in files)
    print(f"Root: {result['root']}")
    print(f"Cutoff: {result['cutoff'].astimezone().isoformat(timespec='seconds')}")
    print(f"{action} {len(files)} files ({total_bytes:,} bytes)")
    for path, size, mtime in files:
        modified = datetime.fromtimestamp(mtime, timezone.utc).astimezone().isoformat(timespec="seconds")
        print(f"{modified}\t{size:>12,}\t{path}")
    if args.delete and result["errors"]:
        for path, error in result["errors"]:
            print(f"Could not remove {path}: {error}")
        return 1
    if not args.delete:
        print("Preview only. Pass --delete to remove these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Find byte-identical self-play training arrays and optionally remove duplicates.

Algorithm: sort replays with complete SGF pairs first, then newest first; use
each NPZ member's CRC and uncompressed size only to form candidate buckets;
compare every training-array member byte for byte before declaring a duplicate.
The random game_id and other metadata do not define training-array identity.
Keep one replay per exact group and write a manifest before deleting any files.
Run --delete only after pausing all readers and writers of the replay directory.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np

from transformer_go.data import _same_training_arrays, _training_signature


DEFAULT_REPLAY_DIR = Path(__file__).resolve().parents[1] / "runs" / "zero-9x9" / "replay"


@dataclass(frozen=True)
class ReplayFile:
    path: Path
    size: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    has_sgf: bool


def _snapshot(path: Path) -> ReplayFile:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"replay is not a regular file: {path}")
    stat = path.stat()
    return ReplayFile(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                      stat.st_ino, path.with_suffix(".sgf").is_file())


def _unchanged(file: ReplayFile):
    current = _snapshot(file.path)
    if (current.size, current.mtime_ns, current.ctime_ns, current.inode) != (
            file.size, file.mtime_ns, file.ctime_ns, file.inode):
        raise RuntimeError(f"replay changed since scan: {file.path}")


def _metadata(path: Path):
    with np.load(path, allow_pickle=False) as shard:
        metadata = json.loads(str(shard["metadata"]))
    if metadata.get("source") != "selfplay":
        raise ValueError(f"refusing to remove non-selfplay replay: {path}")
    return metadata


def scan_duplicates(replay_dir: Path):
    """Return (snapshot, duplicate/keeper pairs), preserving complete newest copies."""
    replay_dir = Path(replay_dir).expanduser()
    if replay_dir.is_symlink() or not replay_dir.is_dir():
        raise ValueError(f"replay directory is missing or a symlink: {replay_dir}")
    replay_dir = replay_dir.resolve()
    files = [_snapshot(path) for path in replay_dir.glob("*.npz")]
    files.sort(key=lambda f: (f.has_sgf, f.mtime_ns, f.path.name), reverse=True)
    kept_by_signature = {}
    duplicates = []
    metadata = {}
    for file in files:
        signature = _training_signature(file.path)
        candidates = kept_by_signature.setdefault(signature, [])
        keeper = next((other for other in candidates
                       if _same_training_arrays(file.path, other.path)), None)
        if keeper is None:
            candidates.append(file)
            continue
        for item in (file, keeper):
            if item.path not in metadata:
                metadata[item.path] = _metadata(item.path)
        duplicates.append((file, keeper, metadata[file.path], metadata[keeper.path]))
    return files, duplicates


def _write_manifest(path: Path, replay_dir: Path, duplicates):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for file, keeper, meta, kept_meta in duplicates:
            entry = {"discarded": file.path.relative_to(replay_dir).as_posix(),
                     "kept": keeper.path.relative_to(replay_dir).as_posix(),
                     "discarded_seed": meta.get("seed"), "kept_seed": kept_meta.get("seed"),
                     "discarded_model": meta.get("model_version"),
                     "kept_model": kept_meta.get("model_version"),
                     "npz_bytes": file.size}
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def delete_duplicates(replay_dir: Path, files, duplicates, manifest: Path):
    """Preflight the complete snapshot, record it, then remove only exact copies."""
    replay_dir = Path(replay_dir).expanduser()
    if replay_dir.is_symlink() or not replay_dir.is_dir():
        raise ValueError(f"replay directory is missing or a symlink: {replay_dir}")
    replay_dir = replay_dir.resolve()
    if {file.path for file in files} != set(replay_dir.glob("*.npz")):
        raise RuntimeError("replay directory changed during scan; retry after pausing training")
    for file in files:
        _unchanged(file)
    cache_dir = replay_dir / ".adjudication-cache"
    if cache_dir.is_symlink():
        raise ValueError(f"refusing to follow cache symlink: {cache_dir}")
    for file, _keeper, _meta, _kept_meta in duplicates:
        for associated in (file.path.with_suffix(".sgf"), cache_dir / file.path.name):
            if associated.is_symlink():
                raise ValueError(f"refusing to follow associated symlink: {associated}")
    manifest = _write_manifest(manifest, replay_dir, duplicates)
    removed = 0
    reclaimed = 0
    for file, keeper, _meta, _kept_meta in duplicates:
        _unchanged(file)
        _unchanged(keeper)
        # The NPZ is removed first; an interrupted cleanup can leave only an
        # orphan SGF, never a shard missing the SGF needed for relabelling.
        file.path.unlink()
        removed += 1
        reclaimed += file.size
        for associated in (file.path.with_suffix(".sgf"), cache_dir / file.path.name):
            if associated.is_file():
                reclaimed += associated.stat().st_size
                associated.unlink()
    return {"removed": removed, "reclaimed_bytes": reclaimed, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--delete", action="store_true", help="delete exact copies after preview")
    parser.add_argument("--manifest", type=Path, help="new JSONL audit file for --delete")
    args = parser.parse_args(argv)
    try:
        files, duplicates = scan_duplicates(args.replay_dir)
        duplicate_bytes = sum(file.size for file, *_ in duplicates)
        print(f"Scanned {len(files)} replay shards; exact duplicates: {len(duplicates)} "
              f"({duplicate_bytes:,} NPZ bytes)")
        if args.delete:
            manifest = args.manifest or (args.replay_dir.parent / "analysis" /
                f"replay-dedupe-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
            result = delete_duplicates(args.replay_dir, files, duplicates, manifest)
            print(f"Removed {result['removed']} replay shards; reclaimed "
                  f"{result['reclaimed_bytes']:,} bytes")
            print(f"Manifest: {result['manifest']}")
        else:
            print("Preview only. Pause training, then pass --delete to remove duplicates.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

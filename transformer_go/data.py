from pathlib import Path
import json
import os
import uuid
import numpy as np
from ._native import Board
from .sgf import parse

SCHEMA_VERSION = 1
ADJUDICATION_VERSION = "katago-boardhistory-v1"
FIELDS = ("features", "legal", "policy", "value", "score", "ownership", "moves_left", "aux_mask")
OLDEST_REPLAY_WEIGHT = 1.0
NEWEST_REPLAY_WEIGHT = 4.0


def validate(data: dict):
    features = data["features"]
    if features.ndim != 4 or features.shape[1] != 21 or features.shape[2] != features.shape[3]:
        raise ValueError("invalid feature schema")
    count, _, n, _ = features.shape
    if count == 0:
        raise ValueError("empty shard")
    shapes = {"legal": (count, n*n+1), "policy": (count, n*n+1), "value": (count,),
              "score": (count,), "ownership": (count, n, n), "moves_left": (count,), "aux_mask": (count,)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid {key}")
    if not np.isfinite(features).all():
        raise ValueError("nonfinite features")
    if not np.isin(data["legal"], [0, 1]).all() or not data["legal"].any(axis=1).all():
        raise ValueError("invalid legal mask")
    p = data["policy"]
    if (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=2e-3) or (p[~data["legal"].astype(bool)] > 0).any():
        raise ValueError("invalid policy target")
    if not np.isin(data["value"], [0, 1, 2]).all() or not np.isin(data["ownership"], [0, 1, 2]).all():
        raise ValueError("invalid class target")
    if not np.isin(data["aux_mask"], [0, 1]).all() or (data["moves_left"] < 0).any():
        raise ValueError("invalid auxiliary targets")


def write_shard(directory: str | Path, data: dict, metadata: dict) -> Path:
    validate(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    game_id = str(uuid.uuid4())
    metadata = {**metadata, "schema_version": SCHEMA_VERSION,
                "adjudication_version": metadata.get("adjudication_version", ADJUDICATION_VERSION),
                "game_id": game_id}
    path = directory / f"{game_id}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data, metadata=np.asarray(json.dumps(metadata)))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def read_shard(path: str | Path):
    with np.load(path, allow_pickle=False) as shard:
        data = {k: shard[k] for k in FIELDS}
        meta = json.loads(str(shard["metadata"]))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported shard schema")
    validate(data)
    return data, meta


def _relabel_legacy_shard(path: Path, data: dict, metadata: dict) -> tuple[dict, dict]:
    """Replay an old self-play SGF and cache corrected KataGo final targets.

    Old samples keep their states and MCTS policies; only outcome-dependent
    labels (WDL, score, ownership) change. The source NPZ/SGF are never edited.
    """
    cache = path.parent / ".adjudication-cache" / path.name
    if cache.exists():
        cached_data, cached_meta = read_shard(cache)
        if (cached_meta.get("game_id") == metadata.get("game_id") and
                cached_meta.get("adjudication_version") == ADJUDICATION_VERSION):
            return cached_data, cached_meta

    sgf_path = path.with_suffix(".sgf")
    if not sgf_path.exists():
        raise ValueError(
            f"legacy replay shard {path} has no matching SGF; cannot recompute KataGo final labels"
        )
    nodes = parse(sgf_path.read_text())
    root = nodes[0]
    n = int(root.get("SZ", [metadata.get("board_size", 0)])[0])
    komi = float(root.get("KM", [metadata.get("komi", 7.5)])[0])
    # The historical SGF writer used RU[Chinese] even when the engine was
    # configured for positional superko. Prefer shard metadata; when absent,
    # retain the original engine default rather than trusting that stale RU.
    rules = metadata.get("rules") or "chinese-area-psk"
    moves = []
    first_player = 1
    for node in nodes[1:]:
        if "B" in node:
            moves.append((1, node["B"][0]))
        elif "W" in node:
            moves.append((-1, node["W"][0]))
        else:
            continue
    if moves and moves[0][0] == -1:
        first_player = -1
    board = Board(n, komi, rules) if first_player == 1 else Board.from_position(
        n, komi, [0] * (n * n), to_play=-1, rules=rules
    )
    actions = []
    for player, coordinate in moves:
        if board.to_play != player:
            raise ValueError(f"SGF turn order mismatch in legacy replay {sgf_path}")
        if not coordinate or coordinate.lower() == "tt":
            action = n * n
        else:
            if len(coordinate) != 2 or not coordinate.isalpha():
                raise ValueError(f"unsupported SGF move in legacy replay {sgf_path}")
            x, y = ord(coordinate[0].lower()) - ord("a"), ord(coordinate[1].lower()) - ord("a")
            if not (0 <= x < n and 0 <= y < n):
                raise ValueError(f"SGF move outside board in legacy replay {sgf_path}")
            action = y * n + x
        board = board.play(action)
        actions.append(action)
    if len(actions) != len(data["value"]) or not board.terminal():
        raise ValueError(f"legacy replay {path} does not contain a complete replayable terminal game")

    formal = board.formal_result()
    winner = int(formal["winner"])
    score = float(formal["score_black"])
    ownership = np.asarray(formal["ownership"], dtype=np.int8).reshape(n, n)
    black_to_play = data["features"][:, 18, 0, 0] > 0.5
    players = np.where(black_to_play, 1, -1).astype(np.int8)
    corrected = {key: value.copy() for key, value in data.items()}
    corrected["value"] = (winner * players + 1).astype(np.int64)
    corrected["score"] = (score * players / (n * n)).astype(np.float32)
    corrected["ownership"] = np.stack([(ownership * player + 1).astype(np.int64) for player in players])
    corrected_meta = {
        **metadata,
        "adjudication_version": ADJUDICATION_VERSION,
        "adjudication_engine": formal["adjudication_engine"],
        "score_black": score,
        "winner": winner,
        "dead_stones": int(np.asarray(formal["dead_stones"], dtype=np.int8).sum()),
        "relabelled_from_adjudication_version": metadata.get("adjudication_version", "legacy-unversioned"),
    }
    validate(corrected)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, **corrected,
            metadata=np.asarray(json.dumps(corrected_meta, separators=(",", ":"))),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(cache)
    return corrected, corrected_meta


def load_replay(directory: str | Path, size: int, max_positions: int, *,
                return_sampling_weights=False):
    """Load a recent replay window and optional per-position freshness weights.

    Shards are read newest first. The returned weights are constant within a
    shard and linearly decrease from ``NEWEST_REPLAY_WEIGHT`` for the newest
    selected shard to ``OLDEST_REPLAY_WEIGHT`` for the oldest selected shard.
    """
    chunks, count, seen = [], 0, set()
    files = sorted(Path(directory).glob("*.npz"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in files:
        data, meta = read_shard(path)
        if data["features"].shape[-1] != size:
            continue
        if meta.get("source") != "selfplay":
            raise ValueError(f"selfplay-only training rejects non-selfplay shard: {path}")
        if meta["game_id"] in seen:
            continue
        if meta.get("adjudication_version") != ADJUDICATION_VERSION:
            data, meta = _relabel_legacy_shard(path, data, meta)
        seen.add(meta["game_id"])
        remaining = max_positions-count
        chunks.append({k: v[-remaining:] for k, v in data.items()})
        count += len(chunks[-1]["value"])
        if count >= max_positions:
            break
    if not chunks:
        raise ValueError(f"no replay positions for {size}x{size} in {directory}")
    replay = {k: np.concatenate([c[k] for c in chunks]) for k in FIELDS}
    if not return_sampling_weights:
        return replay
    shard_count = len(chunks)
    weights = []
    for rank, chunk in enumerate(chunks):
        progress = rank / max(shard_count - 1, 1)
        weight = NEWEST_REPLAY_WEIGHT + progress * (OLDEST_REPLAY_WEIGHT - NEWEST_REPLAY_WEIGHT)
        weights.append(np.full(len(chunk["value"]), weight, dtype=np.float64))
    return replay, np.concatenate(weights)


def transform_spatial(x, symmetry: int):
    result = np.rot90(x, symmetry % 4, axes=(-2, -1))
    return np.flip(result, axis=-1) if symmetry >= 4 else result


def augment(batch: dict, rng):
    batch = {k: v.copy() for k, v in batch.items()}
    n = batch["features"].shape[-1]
    for i in range(len(batch["value"])):
        symmetry = int(rng.integers(8))
        batch["features"][i] = transform_spatial(batch["features"][i], symmetry)
        batch["ownership"][i] = transform_spatial(batch["ownership"][i], symmetry)
        for key in ("legal", "policy"):
            batch[key][i, :-1] = transform_spatial(batch[key][i, :-1].reshape(n, n), symmetry).reshape(-1)
    return batch

"""Durable, process-safe self-play seed reservations for one replay directory."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np


MAX_SEED = (1 << 63) - 1
LEDGER_NAME = ".selfplay-seeds.sqlite3"


def _historical_seeds(directory: Path):
    """Import seeds already present before the ledger was introduced."""
    for path in directory.glob("*.npz"):
        with np.load(path, allow_pickle=False) as shard:
            metadata = json.loads(str(shard["metadata"]))
        seed = metadata.get("seed")
        if type(seed) is not int or not 0 <= seed <= MAX_SEED:
            raise ValueError(f"replay shard has no valid seed: {path}")
        yield seed


def reserve_seed_block(directory: str | Path, count: int, minimum_seed: int = 0) -> int:
    """Commit a unique seed block before starting games; unused seeds stay burned.

    All new self-play writers targeting this replay directory must reserve via
    this ledger. The first caller imports existing shard seeds before allocating.
    SQLite serializes concurrent writers, including writers in other processes.
    """
    if type(count) is not int or count < 1:
        raise ValueError("seed reservation count must be a positive integer")
    if type(minimum_seed) is not int or not 0 <= minimum_seed <= MAX_SEED:
        raise ValueError("minimum seed must be a nonnegative 63-bit integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(directory / LEDGER_NAME, timeout=120, isolation_level=None)) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("CREATE TABLE IF NOT EXISTS used_seeds (seed INTEGER PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS ledger_state "
                       "(id INTEGER PRIMARY KEY CHECK (id = 1))")
            if db.execute("SELECT id FROM ledger_state WHERE id = 1").fetchone() is None:
                db.executemany("INSERT OR IGNORE INTO used_seeds(seed) VALUES (?)",
                               ((seed,) for seed in _historical_seeds(directory)))
                db.execute("INSERT INTO ledger_state(id) VALUES (1)")
            largest = db.execute("SELECT MAX(seed) FROM used_seeds").fetchone()[0]
            first = max(minimum_seed, largest + 1 if largest is not None else 0)
            if first > MAX_SEED - count + 1:
                raise OverflowError("self-play seed space exhausted")
            db.executemany("INSERT INTO used_seeds(seed) VALUES (?)",
                           ((seed,) for seed in range(first, first + count)))
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
    return first

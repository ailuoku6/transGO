"""Spawned actors: each owns a native tree and an independent inference model."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from .config import from_dict
from .model import PolicyValueNet
from .runtime import seed_all, select_device
from .checkpoint import load_checkpoint, model_version
from .search import Evaluator
from .selfplay import generate, validate_min_moves


_ACTOR_STATE = None


def _init_actor(raw, checkpoint, requested_device=None):
    """Load one model per worker process instead of once per game."""
    global _ACTOR_STATE
    config = from_dict(raw)
    seed_all(config.seed)
    requested = config.device if requested_device is None else str(requested_device)
    device = select_device(requested)
    if checkpoint:
        model, config, _ = load_checkpoint(checkpoint, device, selfplay_only=True)
        version = model_version(checkpoint)
    else:
        model = PolicyValueNet(config.model).to(device)
        version = "untrained"
    _ACTOR_STATE = (config, Evaluator(model, device), version)


def _actor_game_task(output, seed, min_moves=0, score_utility=0.0):
    if _ACTOR_STATE is None:
        raise RuntimeError("self-play actor was not initialized")
    config, evaluator, version = _ACTOR_STATE
    events = []
    paths = generate(config, evaluator, output, games=1, seed=seed,
                     model_version=version, report=events.append, min_moves=min_moves,
                     score_utility=score_utility)
    return [str(p) for p in paths], events


def actor_task(raw, checkpoint, output, seed, min_moves=0, score_utility=0.0,
               requested_device=None):
    """Run one actor directly; retained as a small public compatibility API."""
    _init_actor(raw, checkpoint, requested_device)
    return _actor_game_task(output, seed, min_moves, score_utility)[0]


def generate_parallel(config, checkpoint, output, games, workers, seed=None, min_moves=0,
                      score_utility=0.0, report=print, device=None):
    """Generate games in spawned processes and report events in the parent.

    Results are returned in seed/submission order, while progress events are
    forwarded as workers finish.  Each process keeps one model and one
    evaluator for its lifetime, so a 64-game run does not reload the
    checkpoint 64 times.
    """
    if games < 1 or workers < 1:
        raise ValueError("games/workers must be positive")
    validate_min_moves(config.board_size, min_moves)
    base_seed = config.seed if seed is None else seed
    worker_count = min(games, workers)
    results = [None] * games
    futures = {}
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_actor,
        initargs=(config.to_dict(), checkpoint, None if device is None else str(device)),
    ) as pool:
        for i in range(games):
            future = pool.submit(_actor_game_task, output, base_seed + i * 10000,
                                 min_moves, score_utility)
            futures[future] = i
        for future in as_completed(futures):
            index = futures[future]
            paths, events = future.result()
            results[index] = paths
            for event in events:
                report(event)
    return [path for paths in results for path in paths]

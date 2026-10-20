"""Spawned CPU search actors sharing one parent inference model."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from .config import from_dict
from .model import PolicyValueNet
from .runtime import seed_all, select_device
from .checkpoint import load_checkpoint, model_version
from .search import Evaluator
from .selfplay import generate, validate_min_moves
from .inference import InferenceServer, worker_evaluators, validate_batching


_ACTOR_STATE = None


def _load_actor(raw, checkpoint, requested_device=None):
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
    return config, Evaluator(model, device), version


def _init_actor(raw, channels, version):
    global _ACTOR_STATE
    _ACTOR_STATE = (from_dict(raw), worker_evaluators(channels, ("selfplay",))[0], version)


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
    global _ACTOR_STATE
    _ACTOR_STATE = _load_actor(raw, checkpoint, requested_device)
    return _actor_game_task(output, seed, min_moves, score_utility)[0]


def generate_parallel(config, checkpoint, output, games, workers, seed=None, min_moves=0,
                      score_utility=0.0, report=print, device=None, *,
                      inference_batch_size=256, inference_wait_ms=2.0):
    """Generate games in spawned processes and report events in the parent.

    Results are returned in seed/submission order, while progress events are
    forwarded as workers finish. The parent loads one model; CPU workers
    send feature arrays to its cross-game inference batching service.
    """
    if games < 1 or workers < 1:
        raise ValueError("games/workers must be positive")
    validate_batching(inference_batch_size, inference_wait_ms)
    validate_min_moves(config.board_size, min_moves)
    base_seed = config.seed if seed is None else seed
    config, evaluator, version = _load_actor(config.to_dict(), checkpoint, device)
    validate_min_moves(config.board_size, min_moves)
    worker_count = min(games, workers)
    results = [None] * games
    futures = {}
    context = multiprocessing.get_context("spawn")
    with InferenceServer({"selfplay": evaluator}, worker_count, context,
                         inference_batch_size, inference_wait_ms) as server, ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_init_actor,
        initargs=(config.to_dict(), server.channels, version),
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
    report({"event": "inference_summary", "phase": "selfplay", "models": server.stats(),
            "max_batch_size": inference_batch_size, "max_wait_ms": inference_wait_ms})
    return [path for paths in results for path in paths]

"""Spawned CPU search actors sharing one parent inference model."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from .config import from_dict
from .model import PolicyValueNet
from .runtime import seed_all, select_inference_device
from .checkpoint import load_checkpoint, model_version
from .search import Evaluator
from .selfplay import generate, validate_min_moves, MAX_ATTEMPTS_PER_GAME
from .inference import InferenceServer, worker_evaluators, validate_batching
from .exploration import validate_profile
from .seed_ledger import reserve_seed_block


_ACTOR_STATE = None


def selfplay_game_seed(base_seed, game_index):
    # A single-game actor may retry up to MAX_ATTEMPTS_PER_GAME times.
    return base_seed + game_index * MAX_ATTEMPTS_PER_GAME


def _load_actor(raw, checkpoint, requested_device=None, evaluator_options=None):
    config = from_dict(raw)
    seed_all(config.seed)
    requested = config.device if requested_device is None else str(requested_device)
    options = evaluator_options or {}
    device = select_inference_device(requested, options.get("backend", "torch"))
    if checkpoint:
        model, config, _ = load_checkpoint(checkpoint, device, selfplay_only=True)
        version = model_version(checkpoint)
    else:
        model = PolicyValueNet(config.model).to(device)
        version = "untrained"
    evaluator = Evaluator(model, device, **options)
    evaluator.prepare(config.board_size)
    return config, evaluator, version


def _init_actor(raw, channels, version):
    global _ACTOR_STATE
    _ACTOR_STATE = (from_dict(raw), worker_evaluators(channels, ("selfplay",))[0], version)


def _actor_game_task(output, seed, min_moves=0, score_utility=0.0, selfplay_policy="legacy",
                     seeds_reserved=False):
    # Older already-running parent processes submit only four arguments.
    # New parents always pass an explicit profile, including the new default.
    if _ACTOR_STATE is None:
        raise RuntimeError("self-play actor was not initialized")
    config, evaluator, version = _ACTOR_STATE
    events = []
    paths = generate(config, evaluator, output, games=1, seed=seed,
                     model_version=version, report=events.append, min_moves=min_moves,
                     score_utility=score_utility, selfplay_policy=selfplay_policy,
                     _seeds_reserved=seeds_reserved)
    return [str(p) for p in paths], events


def actor_task(raw, checkpoint, output, seed, min_moves=0, score_utility=0.0,
               requested_device=None, selfplay_policy="katago", *, evaluator_options=None):
    """Run one actor directly; retained as a small public compatibility API."""
    global _ACTOR_STATE
    _ACTOR_STATE = _load_actor(raw, checkpoint, requested_device, evaluator_options)
    return _actor_game_task(output, seed, min_moves, score_utility, selfplay_policy)[0]


def generate_parallel(config, checkpoint, output, games, workers, seed=None, min_moves=0,
                      score_utility=0.0, report=print, device=None, *,
                      inference_batch_size=256, inference_wait_ms=2.0, selfplay_policy="katago",
                      evaluator_options=None):
    """Generate games in spawned processes and report events in the parent.

    Results are returned in seed/submission order, while progress events are
    forwarded as workers finish. The parent loads one model; CPU workers
    send feature arrays to its cross-game inference batching service.
    """
    if games < 1 or workers < 1:
        raise ValueError("games/workers must be positive")
    validate_batching(inference_batch_size, inference_wait_ms)
    validate_profile(selfplay_policy)
    validate_min_moves(config.board_size, min_moves)
    requested_seed = config.seed if seed is None else seed
    config, evaluator, version = _load_actor(config.to_dict(), checkpoint, device, evaluator_options)
    validate_min_moves(config.board_size, min_moves)
    base_seed = reserve_seed_block(output, games * MAX_ATTEMPTS_PER_GAME, requested_seed)
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
            future = pool.submit(_actor_game_task, output, selfplay_game_seed(base_seed, i),
                                 min_moves, score_utility, selfplay_policy, True)
            futures[future] = i
        for future in as_completed(futures):
            index = futures[future]
            paths, events = future.result()
            results[index] = paths
            for event in events:
                report(event)
    report({"event": "inference_summary", "phase": "selfplay", "models": server.stats(),
            "backend": evaluator.backend,
            "max_batch_size": inference_batch_size, "max_wait_ms": inference_wait_ms})
    return [path for paths in results for path in paths]

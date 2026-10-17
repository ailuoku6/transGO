"""Spawned actors: each owns a native tree and an independent inference model."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from .config import from_dict
from .model import PolicyValueNet
from .runtime import seed_all, select_device
from .checkpoint import load_checkpoint, model_version
from .search import Evaluator
from .selfplay import generate, validate_min_moves


def actor_task(raw, checkpoint, output, seed, min_moves=0, score_utility=0.0):
    config = from_dict(raw)
    seed_all(config.seed)
    device = select_device(config.device)
    if checkpoint:
        model, config, _ = load_checkpoint(checkpoint, device, selfplay_only=True)
        version = model_version(checkpoint)
    else:
        model = PolicyValueNet(config.model).to(device)
        version = "untrained"
    paths = generate(config, Evaluator(model, device), output, games=1, seed=seed,
                     model_version=version, report=lambda _: None, min_moves=min_moves,
                     score_utility=score_utility)
    return [str(p) for p in paths]


def generate_parallel(config, checkpoint, output, games, workers, seed=None, min_moves=0, score_utility=0.0):
    if games < 1 or workers < 1: raise ValueError("games/workers must be positive")
    validate_min_moves(config.board_size, min_moves)
    base_seed = config.seed if seed is None else seed
    with ProcessPoolExecutor(max_workers=min(games, workers), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(actor_task, config.to_dict(), checkpoint, output, base_seed+i*10000,
                               min_moves, score_utility) for i in range(games)]
        return [path for future in futures for path in future.result()]

"""Reproducible CPU/MPS/CUDA inference and subtree-reuse measurements.

Run from the repository root with python -m benchmarks.search_optimizations.
This measures engineering throughput on fixed move sequences, not strength.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time

import numpy as np
import torch

from transformer_go._native import Board
from transformer_go.config import from_dict, load_config
from transformer_go.inference import InferenceServer, worker_evaluators
from transformer_go.model import PolicyValueNet
from transformer_go.runtime import seed_all, select_device, synchronize
from transformer_go.search import Evaluator, SearchSession, search


_STATE = None


def make_evaluator(config, device):
    seed_all(config.seed)
    return Evaluator(PolicyValueNet(config.model).to(device), device)


def init_worker(raw, requested_device, channels, barrier):
    global _STATE
    config = from_dict(raw)
    evaluator = (worker_evaluators(channels, ("benchmark",))[0] if channels is not None
                 else make_evaluator(config, select_device(requested_device)))
    _STATE = config, evaluator, barrier


def search_sequence(config, evaluator, actions, reuse):
    board = Board(config.board_size, config.komi, config.rules)
    session = SearchSession() if reuse else None
    reused = new = 0
    for move, action in enumerate(actions):
        result = search(board, evaluator, config.search, seed=config.seed + move, session=session)
        reused += result["reused_simulations"]
        new += result["new_simulations"]
        board = board.play(action)
        if session is not None:
            session.advance(action)
    return {"new_simulations": new, "reused_simulations": reused}


def worker_sequence(actions):
    config, evaluator, barrier = _STATE
    evaluator([Board(config.board_size, config.komi, config.rules)])
    baseline_calls, baseline_positions = evaluator.calls, evaluator.positions
    barrier.wait(timeout=60)
    start = time.perf_counter()
    result = search_sequence(config, evaluator, actions, False)
    result.update(seconds=time.perf_counter() - start,
                  requests=evaluator.calls - baseline_calls, positions=evaluator.positions - baseline_positions)
    return result


def make_actions(config, evaluator, moves):
    board = Board(config.board_size, config.komi, config.rules)
    actions = []
    for move in range(moves):
        if board.terminal():
            break
        result = search(board, evaluator, config.search, seed=config.seed + move)
        action = int(np.argmax(result["visits"]))
        actions.append(action)
        board = board.play(action)
    return actions


def parallel_measure(config, device, actions, workers, central, batch_size, wait_ms):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(workers)
    started = time.perf_counter()
    if central:
        evaluator = make_evaluator(config, device)
        with InferenceServer({"benchmark": evaluator}, workers, context, batch_size, wait_ms) as server:
            with ProcessPoolExecutor(workers, mp_context=context, initializer=init_worker,
                                     initargs=(config.to_dict(), str(device), server.channels, barrier)) as pool:
                results = list(pool.map(worker_sequence, [actions] * workers))
        stats = server.stats()["benchmark"]
    else:
        with ProcessPoolExecutor(workers, mp_context=context, initializer=init_worker,
                                 initargs=(config.to_dict(), str(device), None, barrier)) as pool:
            results = list(pool.map(worker_sequence, [actions] * workers))
        requests = sum(r["requests"] for r in results) + workers
        positions = sum(r["positions"] for r in results) + workers
        stats = {"requests": requests, "calls": requests, "positions": positions,
                 "mean_batch_size": positions / requests, "cross_client_batches": 0}
    return {"total_seconds_with_startup": time.perf_counter() - started,
            "search_seconds": max(r["seconds"] for r in results),
            "inference_including_one_warmup_per_worker": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--moves", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--inference-wait-ms", type=float, default=2)
    args = parser.parse_args()
    config = load_config(args.config)
    config.board_size = args.board_size
    config.search.simulations = args.simulations
    device = select_device(args.device)
    evaluator = make_evaluator(config, device)
    actions = make_actions(config, evaluator, args.moves)
    print(json.dumps({"settings": vars(args), "moves_measured": len(actions),
                      "torch": torch.__version__, "device": str(device)}), flush=True)
    for reuse in (False, True):
        evaluator.calls = evaluator.positions = 0
        synchronize(device)
        start = time.perf_counter()
        result = search_sequence(config, evaluator, actions, reuse)
        synchronize(device)
        result.update(seconds=time.perf_counter() - start, calls=evaluator.calls, positions=evaluator.positions)
        print(json.dumps({"tree_reuse": reuse, **result}), flush=True)
    del evaluator
    for central in (False, True):
        result = parallel_measure(config, device, actions, args.workers, central,
                                  args.inference_batch_size, args.inference_wait_ms)
        print(json.dumps({"central_inference": central, **result}), flush=True)


if __name__ == "__main__":
    main()

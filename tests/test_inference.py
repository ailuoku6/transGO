from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import threading
import time

import numpy as np
import pytest
import torch

from transformer_go._native import Board
from transformer_go.checkpoint import save_checkpoint
from transformer_go.config import load_config
from transformer_go.evaluate import evaluate_parallel
from transformer_go.inference import InferenceServer, RemoteEvaluator, worker_evaluators, validate_batching
from transformer_go.model import PolicyValueNet


class ArrayEvaluator:
    def __init__(self, offset=0):
        self.offset = offset
        self.batches = []
        self.fail = False

    def evaluate_arrays(self, features, legal):
        if self.fail:
            raise ValueError("injected model failure")
        self.batches.append(len(features))
        values = features[:, 0, 0, 0] + self.offset
        return (np.repeat(values[:, None], legal.shape[1], axis=1), values, -values)


def arrays(values, size=3):
    features = np.zeros((len(values), 21, size, size), np.float32)
    features[:, 0, 0, 0] = values
    return features, np.ones((len(values), size * size + 1), bool)


def client(server, index, key="model", timeout=5):
    return RemoteEvaluator(server.requests, server.responses[index], server.stopped, index,
                           key, server.max_batch_size, timeout)


def test_cross_client_batching_routes_variable_requests_and_separates_models():
    a, b = ArrayEvaluator(), ArrayEvaluator(100)
    server = InferenceServer({"a": a, "b": b}, 3, multiprocessing.get_context("spawn"),
                             max_batch_size=8, max_wait_ms=1000)
    with server, ThreadPoolExecutor(3) as pool:
        barrier = threading.Barrier(3)

        def run(index, key, values):
            barrier.wait(timeout=5)
            return client(server, index, key).evaluate_arrays(*arrays(values))

        futures = [pool.submit(run, 0, "a", [1, 2]), pool.submit(run, 1, "a", [3]),
                   pool.submit(run, 2, "b", [4, 5])]
        for future, expected in zip(futures, ([1, 2], [3], [104, 105])):
            policy, values, scores = future.result(timeout=5)
            np.testing.assert_array_equal(values, expected)
            np.testing.assert_array_equal(policy[:, 0], expected)
            np.testing.assert_array_equal(scores, -np.asarray(expected))
    assert a.batches == [3] and b.batches == [2]
    stats = server.stats()
    assert stats["a"]["cross_client_batches"] == 1
    assert stats["a"]["requests"] == 2 and stats["a"]["calls"] == 1
    assert not server.thread.is_alive()


def test_timeout_flush_and_oversized_request_splitting():
    model = ArrayEvaluator()
    server = InferenceServer({"model": model}, 2, multiprocessing.get_context("spawn"),
                             max_batch_size=2, max_wait_ms=5)
    with server:
        started = time.monotonic()
        outputs = client(server, 0).evaluate_arrays(*arrays([1, 2, 3, 4, 5]))
        assert time.monotonic() - started < 2
    np.testing.assert_array_equal(outputs[1], [1, 2, 3, 4, 5])
    assert model.batches == [2, 2, 1]
    assert server.stats()["model"]["positions"] == 5


def test_different_board_geometries_are_never_concatenated():
    model = ArrayEvaluator()
    server = InferenceServer({"model": model}, 2, multiprocessing.get_context("spawn"),
                             max_wait_ms=1000)
    with server, ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(client(server, i).evaluate_arrays, *arrays([i], size))
                   for i, size in enumerate((3, 5))]
        assert [f.result(timeout=5)[0].shape for f in futures] == [(1, 10), (1, 26)]
    assert model.batches == [1, 1]


def test_model_failure_propagates_to_all_clients_and_service_recovers():
    model = ArrayEvaluator()
    model.fail = True
    server = InferenceServer({"model": model}, 2, multiprocessing.get_context("spawn"),
                             max_wait_ms=5)
    with server, ThreadPoolExecutor(2) as pool:
        clients = [client(server, i) for i in range(2)]
        futures = [pool.submit(evaluator.evaluate_arrays, *arrays([1])) for evaluator in clients]
        for future in futures:
            with pytest.raises(RuntimeError, match="injected model failure"):
                future.result(timeout=5)
        model.fail = False
        assert clients[0].evaluate_arrays(*arrays([2]))[1][0] == 2
    with pytest.raises(RuntimeError, match="stopped"):
        clients[0].evaluate_arrays(*arrays([3]))


_REMOTE = None
_BARRIER = None


def init_spawned_client(channels, barrier):
    global _REMOTE, _BARRIER
    _REMOTE = worker_evaluators(channels, ("model",))[0]
    _BARRIER = barrier


def spawned_request(value):
    _BARRIER.wait(timeout=15)
    return _REMOTE.evaluate_arrays(*arrays([value]))[1].tolist()


def test_real_spawned_workers_merge_in_parent():
    context = multiprocessing.get_context("spawn")
    model = ArrayEvaluator()
    server = InferenceServer({"model": model}, 2, context, max_wait_ms=1000)
    with server, ProcessPoolExecutor(2, mp_context=context, initializer=init_spawned_client,
                                     initargs=(server.channels, context.Barrier(2))) as pool:
        futures = [pool.submit(spawned_request, value) for value in (3, 7)]
        assert [future.result(timeout=30) for future in futures] == [[3], [7]]
    assert model.batches == [2]
    assert server.stats()["model"]["cross_client_batches"] == 1


def test_real_parallel_evaluation_uses_central_models(tmp_path):
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, PolicyValueNet(config.model), config)
    result = evaluate_parallel(config, checkpoint, checkpoint, games=4, workers=2,
                               device="cpu", max_moves=2, report=lambda _: None,
                               inference_batch_size=8, inference_wait_ms=5,
                               required_candidate_win_prop=0)
    assert result["played"] == 1 and result["early_stopped"]
    assert result["gating"]["decision"] == "accepted"
    assert result["play_settings"]["inference"] == "central-batched"
    assert set(result["inference"]) == {"candidate", "baseline"}
    assert all(stats["positions"] > 0 for stats in result["inference"].values())
    assert not any(t.name == "central-inference" for t in threading.enumerate())


def test_array_evaluator_matches_board_evaluator():
    from transformer_go.search import Evaluator

    config = load_config("configs/smoke.yaml")
    evaluator = Evaluator(PolicyValueNet(config.model), torch.device("cpu"))
    boards = [Board(3), Board(3).play(0)]
    expected = evaluator(boards)
    actual = evaluator.evaluate_arrays(np.stack([b.features() for b in boards]),
                                        np.asarray([b.legal_mask() for b in boards], bool))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("batch,wait", [(0, 1), (True, 1), (1.5, 1), (4, -1), (4, float("nan"))])
def test_invalid_batching_limits(batch, wait):
    with pytest.raises(ValueError, match="batching"):
        validate_batching(batch, wait)


def test_shutdown_after_an_unread_large_response_does_not_block():
    model = ArrayEvaluator()
    server = InferenceServer({"model": model}, 1, multiprocessing.get_context("spawn"))
    with server:
        features, legal = arrays(list(range(128)), 19)
        server.requests.put((0, 1, "model", features, legal))
        deadline = time.monotonic() + 5
        while not server.stats() and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.stats()["model"]["positions"] == 128
    assert not server.thread.is_alive()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU unavailable in this environment")
def test_mps_central_forward_matches_direct():
    from transformer_go.search import Evaluator

    config = load_config("configs/smoke.yaml")
    evaluator = Evaluator(PolicyValueNet(config.model).to("mps"), torch.device("mps"))
    board = Board(3)
    expected = evaluator([board])
    server = InferenceServer({"model": evaluator}, 1, multiprocessing.get_context("spawn"))
    with server:
        actual = client(server, 0)([board])
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

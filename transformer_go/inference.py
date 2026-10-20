"""Central model inference for spawned CPU search workers.

Each worker has one synchronous request in flight. Only CPU feature/mask
arrays cross process boundaries; boards and trees remain worker-local.
"""
from collections import defaultdict
from itertools import count
import math
from queue import Empty
import threading
import time

import numpy as np


def validate_batching(max_batch_size, max_wait_ms):
    if (not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool)
            or max_batch_size < 1 or not math.isfinite(max_wait_ms) or max_wait_ms < 0):
        raise ValueError("invalid inference batching limits")


class RemoteEvaluator:
    def __init__(self, requests, responses, stopped, client_id, model_key,
                 max_batch_size, timeout=120.0, request_ids=None):
        self.requests, self.responses, self.stopped = requests, responses, stopped
        self.client_id, self.model_key = client_id, model_key
        self.max_batch_size, self.timeout = max_batch_size, timeout
        self.request_ids = count(1) if request_ids is None else request_ids
        self.calls = self.positions = 0

    def __call__(self, boards):
        features = np.stack([board.features() for board in boards])
        legal = np.asarray([board.legal_mask() for board in boards], dtype=bool)
        return self.evaluate_arrays(features, legal)

    def evaluate_arrays(self, features, legal):
        if len(features) == 0 or len(features) != len(legal):
            raise ValueError("inference requires matching nonempty features and legal masks")
        parts = []
        for start in range(0, len(features), self.max_batch_size):
            if self.stopped.is_set():
                raise RuntimeError("central inference service stopped")
            self.calls += 1
            request_id = next(self.request_ids)
            end = start + self.max_batch_size
            self.requests.put((self.client_id, request_id, self.model_key,
                               features[start:end], legal[start:end]))
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    response_key, response_id, outputs, error = self.responses.get(timeout=min(1.0, self.timeout))
                    break
                except Empty:
                    if self.stopped.is_set() or time.monotonic() >= deadline:
                        raise RuntimeError("central inference service stopped or timed out") from None
            if response_key != self.model_key or response_id != request_id:
                raise RuntimeError("central inference response sequence mismatch")
            if error is not None:
                raise RuntimeError(f"central inference failed: {error}")
            self.positions += len(features[start:end])
            parts.append(outputs)
        return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))


def worker_evaluators(channels, model_keys):
    requests, responses, ids, stopped, max_batch_size = channels
    # A dead service must not make worker exit wait for an unread queue pipe.
    requests.cancel_join_thread()
    client_id = ids.get(timeout=30)
    request_ids = count(1)
    return tuple(RemoteEvaluator(requests, responses[client_id], stopped, client_id, key,
                                 max_batch_size, request_ids=request_ids) for key in model_keys)


class InferenceServer:
    """One parent thread owns all forwards, grouped by model and geometry.

    Use outside the process-pool context so inference stays alive until all
    running games have exited, including discarded gatekeeper games.
    """
    def __init__(self, evaluators, workers, context, max_batch_size=256, max_wait_ms=2.0):
        validate_batching(max_batch_size, max_wait_ms)
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("inference workers must be a positive integer")
        self.evaluators = evaluators
        self.workers, self.max_batch_size = workers, max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.requests = context.Queue()
        self.responses = [context.Queue() for _ in range(workers)]
        self.ids = context.Queue()
        for i in range(workers):
            self.ids.put(i)
        self.stopped = context.Event()
        self.thread = threading.Thread(target=self._serve, name="central-inference", daemon=True)
        self.error = None
        self._stats = defaultdict(lambda: {"requests": 0, "calls": 0, "positions": 0,
                                           "max_batch_size": 0, "cross_client_batches": 0})

    @property
    def channels(self):
        return self.requests, self.responses, self.ids, self.stopped, self.max_batch_size

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.requests.put(None)
        self.thread.join()
        for queue in [self.requests, self.ids, *self.responses]:
            # At this point workers have exited. A crashed client may have
            # left an unread response; never wait for that pipe to drain.
            queue.cancel_join_thread()
            queue.close()
        if self.error is not None and exc[0] is None:
            raise RuntimeError(f"central inference server failed: {self.error}")

    def stats(self):
        return {key: {**values, "mean_batch_size": values["positions"] / values["calls"]
                      if values["calls"] else 0.0}
                for key, values in self._stats.items()}

    def _forward(self, requests):
        try:
            key = requests[0][2]
            features = np.concatenate([request[3] for request in requests])
            legal = np.concatenate([request[4] for request in requests])
            outputs = self.evaluators[key].evaluate_arrays(features, legal)
            if len(outputs) != 3 or any(len(value) != len(features) for value in outputs):
                raise ValueError("invalid central evaluator output shapes")
            stats = self._stats[key]
            stats["requests"] += len(requests)
            stats["calls"] += 1
            stats["positions"] += len(features)
            stats["max_batch_size"] = max(stats["max_batch_size"], len(features))
            stats["cross_client_batches"] += len({request[0] for request in requests}) > 1
            offset = 0
            for client_id, request_id, _, batch, _ in requests:
                count = len(batch)
                result = tuple(np.asarray(value[offset:offset + count], dtype=np.float32)
                               for value in outputs)
                self.responses[client_id].put((key, request_id, result, None))
                offset += count
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            for client_id, request_id, key, *_ in requests:
                self.responses[client_id].put((key, request_id, None, message))

    def _serve(self):
        try:
            while not self.stopped.is_set():
                first = self.requests.get()
                if first is None:
                    break
                pending = [first]
                deadline = time.monotonic() + self.max_wait
                # Bound collection by the number of clients, not total rows:
                # a full single-client batch should still drain peers fairly.
                while len(pending) < self.workers:
                    try:
                        remaining = deadline - time.monotonic()
                        request = (self.requests.get(timeout=remaining) if remaining > 0
                                   else self.requests.get_nowait())
                    except Empty:
                        break
                    if request is None:
                        break
                    pending.append(request)
                groups = defaultdict(list)
                for request in pending:
                    _, _, key, features, legal = request
                    groups[(key, features.shape[1:], legal.shape[1:])].append(request)
                for requests in groups.values():
                    batch, size = [], 0
                    for request in requests:
                        count = len(request[3])
                        if batch and size + count > self.max_batch_size:
                            self._forward(batch)
                            batch, size = [], 0
                        batch.append(request)
                        size += count
                    if batch:
                        self._forward(batch)
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.stopped.set()

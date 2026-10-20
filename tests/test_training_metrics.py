import csv
import importlib
import json

import numpy as np
import pytest
import torch

from transformer_go.config import load_config
from transformer_go.model import PolicyValueNet
from transformer_go.selfplay import generate
from transformer_go.training_metrics import LOSS_KEYS, TrainingLog, append_loss_history


def test_loss_history_preserves_repeated_iterations_and_flushed_steps(tmp_path):
    summaries = []
    for _ in range(2):
        log = TrainingLog(tmp_path / "training", tmp_path / "candidate.pt", 10, 2, iteration=7)
        try:
            for step, loss in ((11, 4.0), (12, 2.0)):
                metrics = {key: loss for key in LOSS_KEYS}
                metrics.update(step=step, positions=42, device="cpu")
                log.record(metrics)
                # File is readable before close, including the just-completed step.
                assert json.loads(log.path.read_text().splitlines()[-1])["step"] == step
            summary = log.finish(metrics, {"policy": 1.0})
            assert summary["mean"] == {key: 3.0 for key in LOSS_KEYS}
            assert summary["min"] == summary["last"] == {key: 2.0 for key in LOSS_KEYS}
            assert summary["max"] == {key: 4.0 for key in LOSS_KEYS}
            assert summary["start_step"] == 11 and summary["end_step"] == 12
            assert json.loads(log.summary_path.read_text()) == summary
            append_loss_history(tmp_path / "training-loss.csv", summary)
            summaries.append(summary)
        finally:
            log.close()
    assert summaries[0]["summary_path"] != summaries[1]["summary_path"]
    with (tmp_path / "training-loss.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(row["iteration"] == "7" for row in rows)
    assert [row["summary_path"] for row in rows] == [s["summary_path"] for s in summaries]


def test_failed_training_keeps_completed_step_log_without_completed_summary(tmp_path, monkeypatch):
    module = importlib.import_module("transformer_go.train")
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    config.search.dirichlet_epsilon = 0

    def pass_only(boards):
        policy = np.zeros((len(boards), 10), dtype=np.float32)
        policy[:, -1] = 1
        return policy, np.zeros(len(boards), dtype=np.float32)

    generate(config, pass_only, tmp_path / "replay", report=lambda _: None)
    original_losses = module.losses
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("injected second-step failure")
        return original_losses(*args)

    monkeypatch.setattr(module, "losses", fail_second)
    directory = tmp_path / "custom-metrics"
    output = tmp_path / "model.pt"
    with pytest.raises(FloatingPointError, match="second-step failure"):
        module.train(config, PolicyValueNet(config.model), torch.device("cpu"),
                     tmp_path / "replay", output, 3, report=lambda _: None,
                     metrics_dir=directory, iteration=4)
    logs = list(directory.glob("steps-*/steps.jsonl"))
    assert len(logs) == 1 and not output.exists()
    step = json.loads(logs[0].read_text())
    assert step["step"] == 1 and step["iteration"] == 4
    assert not list(directory.glob("steps-*/summary.json"))
    assert all(np.isfinite(step[key]) for key in LOSS_KEYS)

"""Durable loss records, separate from model configuration and checkpoints."""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp


LOSS_KEYS = ("total", "policy", "value", "score", "ownership", "moves", "grad_norm")
LOSS_FUNCTION_VERSION = 2


class TrainingLog:
    def __init__(self, directory, output, start_step, steps, iteration=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(mkdtemp(prefix=f"steps-{start_step + 1}-", dir=directory))
        self.path = self.directory / "steps.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.stream = self.path.open("w", encoding="utf-8")
        self.header = {"schema_version": 1, "iteration": iteration,
                       "loss_version": LOSS_FUNCTION_VERSION,
                       "checkpoint": str(output), "start_step": start_step + 1,
                       "requested_steps": steps,
                       "started_at": datetime.now(timezone.utc).isoformat()}
        self.count = 0
        self.sums = {key: 0.0 for key in LOSS_KEYS}
        self.minimum = {key: float("inf") for key in LOSS_KEYS}
        self.maximum = {key: -float("inf") for key in LOSS_KEYS}

    def record(self, metrics):
        self.stream.write(json.dumps({"event": "train_step", "iteration": self.header["iteration"],
                                      **metrics}, allow_nan=False) + "\n")
        # Preserve completed steps even if training/evaluation is interrupted.
        self.stream.flush()
        self.count += 1
        for key in LOSS_KEYS:
            value = metrics[key]
            self.sums[key] += value
            self.minimum[key] = min(self.minimum[key], value)
            self.maximum[key] = max(self.maximum[key], value)

    def finish(self, metrics, loss_weights):
        summary = {**self.header, "event": "training_summary", "status": "completed",
                   "finished_at": datetime.now(timezone.utc).isoformat(),
                   "end_step": metrics["step"], "steps": self.count,
                   "positions": metrics["positions"], "device": metrics["device"],
                   "loss_weights": loss_weights,
                   "mean": {key: value / self.count for key, value in self.sums.items()},
                   "min": self.minimum, "max": self.maximum,
                   "last": {key: metrics[key] for key in LOSS_KEYS},
                   "metrics_path": str(self.path), "summary_path": str(self.summary_path)}
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.summary_path)
        return summary

    def close(self):
        self.stream.close()


def append_loss_history(path, summary):
    """One row per completed training attempt; repeated iterations stay distinct."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: summary[key] for key in
           ("iteration", "start_step", "end_step", "steps", "positions", "finished_at")}
    row["loss_version"] = summary.get("loss_version", 1)
    for aggregate in ("mean", "last"):
        row.update({f"{aggregate}_{key}": summary[aggregate][key] for key in LOSS_KEYS})
    row.update(checkpoint=summary["checkpoint"], summary_path=summary["summary_path"])
    needs_header = not path.exists() or path.stat().st_size == 0
    old_rows, old_fields = [], []
    if not needs_header:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
    fields = old_fields + [key for key in row if key not in old_fields]
    if old_fields and fields != old_fields:
        with NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    else:
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

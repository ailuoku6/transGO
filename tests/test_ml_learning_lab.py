import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ml_learning_lab import demo_attention, demo_gradient, demo_probabilities


def test_beginner_numeric_examples_match_the_lessons():
    result = demo_gradient()
    assert result["loss_before"] == 16
    assert result["gradient"] == -16
    assert result["weight_after"] == pytest.approx(2.6)
    assert result["loss_after"] == pytest.approx(0.64)
    probability = demo_probabilities()
    assert probability["masked_probability"][1] == 0
    assert probability["masked_sum"] == pytest.approx(1)
    assert probability["value"] == pytest.approx(0.5)
    attention = demo_attention()
    assert attention["weighted_information"][0][0] == pytest.approx(12.689414, abs=1e-5)
    assert attention["masked_information"][0][0] == 10


def test_all_learning_experiments_work_outside_project_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/ml_learning_lab.py"), "--lesson", "all"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=True,
    )
    decoder = json.JSONDecoder()
    text = completed.stdout.lstrip()
    records = []
    while text:
        record, offset = decoder.raw_decode(text)
        records.append(record)
        text = text[offset:].lstrip()
    results = {r["lesson"]: r for r in records}
    assert len(results) == 9
    assert results["tensors"]["batch_shape"] == [1, 21, 3, 3]
    assert results["tensors"]["opponent_stone_at_0"] == 1
    assert results["forward"]["parameters"] == 41610
    assert results["forward"]["output_shapes"]["ownership_logits"] == [1, 3, 3, 3]
    assert results["losses"]["finite_gradient"]
    assert results["losses"]["illegal_gradient"] == 0
    assert results["replay"]["legal_policy_aligned"]
    training = results["train"]
    assert training["games"] == 2 and training["positions"] > 0
    assert 0 < training["training_pool_positions"] <= training["positions"]
    assert training["first_record_targets"]["policy_sum"] == pytest.approx(1)
    assert training["weight_changed"] and training["max_weight_change"] > 0
    assert training["saved_step"] == 2 and training["resumed_step"] == 3
    assert training["logged_steps"] == 3 and training["loss_version"] == 2
    assert training["temporary_files_removed"]
    assert not list(tmp_path.iterdir())

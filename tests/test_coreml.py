"""Core ML graph parity, evaluator contracts and optional hardware integration."""
from copy import deepcopy
import importlib.util
import os
import platform
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformer_go import coreml
from transformer_go._native import Board
from transformer_go.config import ModelConfig, load_config
from transformer_go.model import PolicyValueNet
from transformer_go.runtime import select_inference_device
from transformer_go.search import Evaluator, analyze_position


@pytest.mark.parametrize("size", [2, 3, 9, 13, 19])
@pytest.mark.parametrize("global_every", [1, 2, 3])
def test_export_graph_matches_every_pytorch_head(size, global_every):
    torch.manual_seed(17)
    model = PolicyValueNet(ModelConfig(width=16, layers=2, heads=4, window=3,
                                       global_every=global_every)).eval()
    features = torch.randn(2, 21, size, size)
    with torch.inference_mode():
        expected = model(features)
        actual = coreml.ExportNetwork(model, size, 2).eval()(features)
    for name, value in zip(coreml.OUTPUT_NAMES, actual):
        torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=2e-6)


@pytest.fixture
def stub_coreml(monkeypatch):
    """Run the real export graph behind the runtime boundary, without Apple HW."""
    monkeypatch.setattr(coreml, "require_coreml", lambda: SimpleNamespace(__version__="test"))
    calls = []

    def load(backend, size):
        if size not in backend.models:
            network = coreml.ExportNetwork(backend.model, size, backend.batch_size).eval()

            def predict(inputs):
                features = inputs["features"]
                calls.append(features.copy())
                with torch.inference_mode():
                    outputs = network(torch.from_numpy(features))
                return {name: value.numpy() for name, value in zip(coreml.OUTPUT_NAMES, outputs)}

            backend.models[size] = SimpleNamespace(predict=predict)
        return backend.models[size]

    monkeypatch.setattr(coreml.CoreMLBackend, "_load_model", load)
    return calls


@pytest.mark.parametrize("count", [1, 4, 5, 9])
def test_padding_chunking_masking_and_stats(stub_coreml, count):
    torch.manual_seed(9)
    model = PolicyValueNet(ModelConfig(width=16, layers=2, heads=4))
    boards = [Board(3).play(0)] * count
    features = np.stack([board.features() for board in boards])
    legal = np.asarray([board.legal_mask() for board in boards], dtype=bool)
    legal[-1] = False
    legal[-1, -1] = True
    expected = Evaluator(model, torch.device("cpu")).evaluate_arrays(features, legal)
    evaluator = Evaluator(model, torch.device("cpu"), backend="coreml", coreml_batch_size=4)
    actual = evaluator.evaluate_arrays(features, legal)
    for value, reference in zip(actual, expected):
        assert value.dtype == np.float32
        np.testing.assert_allclose(value, reference, rtol=1e-5, atol=2e-6)
    assert actual[0][-1, -1] == 1
    assert np.all(actual[0][~legal] == 0)
    assert evaluator.calls == 1 and evaluator.positions == count
    assert len(stub_coreml) == (count + 3) // 4
    assert all(batch.shape == (4, 21, 3, 3) for batch in stub_coreml)
    if count % 4:
        assert not stub_coreml[-1][count % 4:].any()


def test_snapshot_preserves_training_model_and_changes_cache_identity(stub_coreml):
    model = PolicyValueNet(ModelConfig(width=16, layers=2, heads=4)).train()
    state = deepcopy(model.state_dict())
    first = coreml.CoreMLBackend(model, batch_size=2)
    second = coreml.CoreMLBackend(model, batch_size=2)
    assert first.fingerprint == second.fingerprint
    assert model.training and all(parameter.requires_grad for parameter in model.parameters())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    with torch.no_grad():
        model.policy.bias.add_(1)
    assert coreml.CoreMLBackend(model, batch_size=2).fingerprint != first.fingerprint
    assert coreml.CoreMLBackend(model, batch_size=4).fingerprint != second.fingerprint
    torch.testing.assert_close(first.model.policy.bias, state["policy.bias"], rtol=0, atol=0)


def test_dynamic_board_geometry_and_analysis_contract(stub_coreml):
    config = load_config("configs/smoke.yaml")
    evaluator = Evaluator(PolicyValueNet(config.model), torch.device("cpu"),
                          backend="coreml", coreml_batch_size=4)
    evaluator.prepare(3)
    assert evaluator.calls == evaluator.positions == 0
    for size in (3, 5, 3):
        policy, value, score = evaluator([Board(size)])
        assert policy.shape == (1, size * size + 1)
        assert value.shape == score.shape == (1,)
    assert set(evaluator.coreml.models) == {3, 5}
    result = analyze_position(Board(3), evaluator, config.search)
    assert len(result["model"]["ownership"]) == 3
    assert len(result["model"]["policy"]) == 10


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected_before_import(size):
    with pytest.raises(ValueError, match="positive integer"):
        coreml.CoreMLBackend(PolicyValueNet(ModelConfig(width=16, layers=1, heads=4)), size)


@pytest.mark.parametrize("shape", [(0, 21, 3, 3), (1, 20, 3, 3), (1, 21, 3, 4),
                                   (1, 21, 20, 20), (21, 3, 3)])
def test_invalid_feature_shapes(stub_coreml, shape):
    backend = coreml.CoreMLBackend(PolicyValueNet(ModelConfig(width=16, layers=1, heads=4)))
    with pytest.raises(ValueError, match="expects nonempty"):
        backend.predict(np.zeros(shape, dtype=np.float32))


def test_non_finite_predictions_fail_explicitly(stub_coreml, monkeypatch):
    backend = coreml.CoreMLBackend(PolicyValueNet(ModelConfig(width=16, layers=1, heads=4)), 1)
    bad = SimpleNamespace(predict=lambda _: {name: np.full((1,), np.nan, dtype=np.float32)
                                            for name in coreml.OUTPUT_NAMES})
    monkeypatch.setattr(backend, "_load_model", lambda _: bad)
    with pytest.raises(RuntimeError, match="non-finite"):
        backend.predict(np.zeros((1, 21, 3, 3), dtype=np.float32))


def test_runtime_failure_has_context_and_no_torch_fallback(stub_coreml, monkeypatch):
    model = PolicyValueNet(ModelConfig(width=16, layers=1, heads=4))
    evaluator = Evaluator(model, torch.device("cpu"), backend="coreml", coreml_batch_size=4)

    def fail(_):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(evaluator.coreml, "_load_model", lambda _: SimpleNamespace(predict=fail))
    with pytest.raises(RuntimeError, match="CPU_AND_NE prediction failed.*3x3.*compiler unavailable"):
        evaluator([Board(3)])
    assert evaluator.calls == evaluator.positions == 0


def test_compute_plan_reports_static_preferences_and_nested_operations(stub_coreml, monkeypatch):
    backend = coreml.CoreMLBackend(PolicyValueNet(ModelConfig(width=16, layers=1, heads=4)))
    device = type("MLNeuralEngineComputeDevice", (), {})()
    leaf = SimpleNamespace(blocks=[])
    parent = SimpleNamespace(blocks=[SimpleNamespace(operations=[leaf])])
    plan = SimpleNamespace(
        model_structure=SimpleNamespace(program=SimpleNamespace(functions={
            "main": SimpleNamespace(block=SimpleNamespace(operations=[parent]))})),
        get_compute_device_usage_for_mlprogram_operation=lambda op: (
            SimpleNamespace(preferred_compute_device=device) if op is parent else None))
    backend.ct = SimpleNamespace(
        ComputeUnit=SimpleNamespace(CPU_AND_NE="CPU_AND_NE"),
        models=SimpleNamespace(compute_plan=SimpleNamespace(MLComputePlan=SimpleNamespace(
            load_from_path=lambda *args, **kwargs: plan))))
    monkeypatch.setattr(backend, "_load_model", lambda _: SimpleNamespace(get_compiled_model_path=lambda: "model.mlmodelc"))
    report = backend.compute_plan_report(3)
    assert report["preferred_operations"] == {"neural_engine": 1, "unclassified": 1}
    assert report["evidence"] == "static_compute_plan" and not report["runtime_utilization_measured"]


def test_unavailable_compute_plan_is_a_diagnostic_not_a_prediction_failure(stub_coreml, monkeypatch):
    backend = coreml.CoreMLBackend(PolicyValueNet(ModelConfig(width=16, layers=1, heads=4)))

    def fail(_):
        raise RuntimeError("compute plan unavailable")

    monkeypatch.setattr(backend, "_load_model", fail)
    assert backend.compute_plan_report(3) == {
        "available": False, "reason": "compute plan unavailable", "runtime_utilization_measured": False}


def test_coreml_loads_source_on_cpu_without_requiring_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_inference_device("mps", "coreml") == torch.device("cpu")
    assert select_inference_device("cpu", "torch") == torch.device("cpu")
    with pytest.raises(ValueError, match="backend"):
        select_inference_device("auto", "unknown")


def test_unsupported_platform_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(coreml, "coreml_report", lambda: {"hardware_eligible": False})
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        coreml.require_coreml()


def test_missing_optional_dependency_has_install_instructions(monkeypatch):
    monkeypatch.setattr(coreml, "coreml_report", lambda: {"hardware_eligible": True})
    monkeypatch.setitem(sys.modules, "coremltools", None)
    with pytest.raises(RuntimeError, match=r"pip install.*coreml"):
        coreml.require_coreml()


@pytest.mark.parametrize("command,required", [
    ("selfplay", []), ("eval", ["--candidate", "new.pt", "--baseline", "old.pt"]),
    ("loop", []), ("gtp", []), ("json-rpc", []), ("play", []), ("benchmark", []),
])
def test_all_inference_commands_expose_backend_options(monkeypatch, command, required):
    from transformer_go import cli
    captured = []
    name = "cmd_rpc" if command == "json-rpc" else f"cmd_{command}"
    monkeypatch.setattr(cli, name, captured.append)
    monkeypatch.setattr(sys, "argv", ["transformer-go", command, *required,
                                      "--inference-backend", "coreml", "--coreml-batch-size", "4"])
    cli.main()
    assert cli.evaluator_options(captured[0]) == {
        "backend": "coreml", "coreml_batch_size": 4, "coreml_cache_dir": "runs/coreml"}


def test_parallel_selfplay_and_evaluation_use_parent_coreml(stub_coreml, tmp_path):
    from transformer_go.actors import generate_parallel
    from transformer_go.checkpoint import save_checkpoint
    from transformer_go.evaluate import evaluate_parallel
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, PolicyValueNet(config.model), config)
    options = {"backend": "coreml", "coreml_batch_size": 4}
    events = []
    paths = generate_parallel(config, checkpoint, tmp_path / "replay", 2, 2,
                              evaluator_options=options, report=events.append)
    assert len(paths) == 2 and stub_coreml
    assert events[-1]["backend"] == "coreml"
    result = evaluate_parallel(config, checkpoint, checkpoint, games=2, workers=2,
                               max_moves=8, evaluator_options=options, report=lambda _: None)
    assert result["played"] == 2
    assert result["play_settings"]["inference_backend"] == "coreml"


def test_loop_keeps_training_in_pytorch_and_records_backend(stub_coreml, tmp_path):
    from transformer_go.pipeline import run_pipeline
    import json
    config = load_config("configs/smoke.yaml")
    config.device = "cpu"
    run_pipeline(config, torch.device("cpu"), tmp_path, iterations=1, selfplay_games=1,
                 train_steps=1, eval_games=2, evaluator_options={"backend": "coreml", "coreml_batch_size": 4},
                 report=lambda _: None)
    record = json.loads((tmp_path / "iteration-0001.json").read_text())
    assert record["inference_backend"] == "coreml" and record["coreml_batch_size"] == 4
    assert record["training"]["device"] == "cpu"
    assert (tmp_path / "learner.pt").is_file() and stub_coreml


@pytest.mark.skipif(
    os.environ.get("TRANSFORMER_GO_TEST_COREML") != "1" or platform.system() != "Darwin"
    or platform.machine() != "arm64" or importlib.util.find_spec("coremltools") is None,
    reason="opt-in Apple Silicon runtime test: TRANSFORMER_GO_TEST_COREML=1",
)
@pytest.mark.parametrize("size", [3, 9, 19])
def test_real_coreml_conversion_prediction_and_cache(tmp_path, monkeypatch, size):
    torch.manual_seed(15)
    model = PolicyValueNet(ModelConfig(width=32, layers=2, heads=4, window=3)).eval()
    boards = [Board(size).play(0)] * 5
    reference = Evaluator(model, torch.device("cpu"))(boards)
    options = {"backend": "coreml", "coreml_batch_size": 4, "coreml_cache_dir": tmp_path}
    evaluator = Evaluator(model, torch.device("cpu"), **options)
    actual = evaluator(boards)
    for value, expected in zip(actual, reference):
        np.testing.assert_allclose(value, expected, rtol=0.03, atol=0.003)
    raw, *_ = evaluator.predict(boards)
    expected_raw, *_ = Evaluator(model, torch.device("cpu")).predict(boards)
    for name in coreml.OUTPUT_NAMES:
        # ANE FP16 logits round more than the normalized predictions.
        # Check the actual probabilities with a separate absolute bound.
        torch.testing.assert_close(raw[name], expected_raw[name], rtol=0.03, atol=0.015)
    for name in ("policy_logits", "value_logits", "ownership_logits"):
        torch.testing.assert_close(raw[name].softmax(-1), expected_raw[name].softmax(-1),
                                   rtol=0, atol=0.005)
    packages = list(tmp_path.glob("*.mlpackage"))
    assert len(packages) == 1

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("cached weights/shape should not be converted again")

    import coremltools
    monkeypatch.setattr(coremltools, "convert", unexpected_conversion)
    again = Evaluator(model, torch.device("cpu"), **options)(boards)
    for first, second in zip(actual, again):
        np.testing.assert_array_equal(first, second)

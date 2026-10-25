"""Optional Apple Neural Engine inference via Core ML (CPU_AND_NE).

Only immutable inference snapshots are converted. Training stays in PyTorch;
Core ML decides which supported operations actually execute on the ANE.
"""
from copy import deepcopy
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OUTPUT_NAMES = ("policy_logits", "value_logits", "score_mean", "score_logvar",
                "ownership_logits", "moves_left")
EXPORT_VERSION = 1


def coreml_report():
    """Report eligibility, not evidence of actual Neural Engine execution."""
    try:
        version = importlib.metadata.version("coremltools")
    except importlib.metadata.PackageNotFoundError:
        version = None
    macos = platform.mac_ver()[0]
    eligible = (platform.system() == "Darwin" and platform.machine() == "arm64"
                and bool(macos) and int(macos.split(".")[0]) >= 13)
    return {"installed": version is not None, "version": version,
            "hardware_eligible": eligible, "compute_units": "CPU_AND_NE",
            "ane_execution_verified": False}


def require_coreml():
    status = coreml_report()
    if not status["hardware_eligible"]:
        raise RuntimeError("Core ML Neural Engine inference requires Apple Silicon and macOS 13+")
    try:
        import coremltools as ct
    except ImportError as error:
        raise RuntimeError("Core ML backend requires the optional dependency: "
                           "python -m pip install -e '.[coreml]'") from error
    return ct


class ExportNetwork(nn.Module):
    """Fixed geometry with explicit attention, avoiding fused PyTorch kernels.

    Coordinates and local masks are constants for each exported board size.
    All learned weights and all six prediction heads match PolicyValueNet.
    Legal-move masking is applied on CPU after prediction, outside this graph.
    """
    def __init__(self, model, board_size, batch_size=1):
        super().__init__()
        self.net = deepcopy(model).cpu().float().eval().requires_grad_(False)
        self.board_size = board_size
        self.batch_size = batch_size
        self.tokens = board_size * board_size + 1
        d = self.net.config.width
        self.heads = self.net.config.heads
        self.head_dim = d // self.heads
        axis = torch.linspace(-1, 1, board_size)
        row, col = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack([row, col, (math.pi * row).sin(), (math.pi * row).cos(),
                              (math.pi * col).sin(), (math.pi * col).cos()], dim=-1).reshape(-1, 6)
        self.register_buffer("position", self.net.position(coords).detach())
        axis = torch.arange(board_size)
        row, col = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack([row.flatten(), col.flatten()], dim=1)
        distant = (points[:, None] - points[None, :]).abs().amax(-1) > self.net.config.window // 2
        mask = torch.zeros(self.tokens, self.tokens, dtype=torch.bool)
        mask[:-1, :-1] = distant
        mask[-1, :-1] = True
        mask[:-1, -1] = True
        self.register_buffer("attention_bias", torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, -math.inf))

    def forward(self, features):
        batch = self.batch_size
        x = self.net.stem(features).flatten(2).transpose(1, 2) + self.position
        x = torch.cat([x, self.net.pass_token.expand(batch, -1, -1)], dim=1)
        for index, block in enumerate(self.net.blocks):
            h = block.norm1(x)
            q, k, v = F.linear(h, block.attention.in_proj_weight,
                               block.attention.in_proj_bias).chunk(3, dim=-1)
            q = q.reshape(batch, self.tokens, self.heads, self.head_dim).transpose(1, 2)
            k = k.reshape(batch, self.tokens, self.heads, self.head_dim).transpose(1, 2)
            v = v.reshape(batch, self.tokens, self.heads, self.head_dim).transpose(1, 2)
            weights = (q * (self.head_dim ** -0.5)) @ k.transpose(-2, -1)
            if (index + 1) % self.net.config.global_every:
                weights = weights + self.attention_bias
            h = (weights.softmax(-1) @ v).transpose(1, 2).reshape(batch, self.tokens, -1)
            x = x + block.attention.out_proj(h)
            a, b = block.up(block.norm2(x)).chunk(2, dim=-1)
            x = x + block.down(F.silu(a) * b)
        x = self.net.norm(x)
        pooled = x.mean(1)
        score = self.net.score(pooled)
        return (self.net.policy(x).squeeze(-1), self.net.value(pooled), score[:, 0],
                score[:, 1].clamp(-8, 8),
                self.net.ownership(x[:, :-1]).reshape(batch, self.board_size, self.board_size, 3),
                F.softplus(self.net.moves(pooled).squeeze(-1)))


class CoreMLBackend:
    def __init__(self, model, batch_size=16, cache_dir="runs/coreml"):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("coreml_batch_size must be a positive integer")
        self.ct = require_coreml()
        # Never move or mutate the learner/shared PyTorch module itself.
        self.model = deepcopy(model).cpu().float().eval().requires_grad_(False)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.models = {}
        metadata = {"export_version": EXPORT_VERSION, "model": asdict(model.config),
                    "torch": str(torch.__version__), "coremltools": self.ct.__version__,
                    "batch_size": batch_size}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        self.fingerprint = digest.hexdigest()

    def _load_model(self, board_size):
        if board_size in self.models:
            return self.models[board_size]
        ct = self.ct
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        package = self.cache_dir / f"{self.fingerprint}-{board_size}x{board_size}.mlpackage"
        if not package.exists():
            network = ExportNetwork(self.model, board_size, self.batch_size).eval()
            example = torch.zeros(self.batch_size, self.model.config.channels, board_size, board_size)
            with torch.inference_mode():
                traced = torch.jit.trace(network, example)
            converted = ct.convert(
                traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
                inputs=[ct.TensorType(name="features", shape=example.shape, dtype=np.float32)],
                outputs=[ct.TensorType(name=name, dtype=np.float32) for name in OUTPUT_NAMES],
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True,
            )
            converted.short_description = "Transformer Go inference; CPU and Neural Engine allowed, GPU disabled"
            # Publish only complete packages; concurrent converters may share a cache.
            with TemporaryDirectory(prefix="export-", dir=self.cache_dir) as temporary:
                staged = Path(temporary) / "model.mlpackage"
                converted.save(str(staged))
                try:
                    staged.rename(package)
                except OSError:
                    if not package.is_dir():
                        raise
        try:
            compiled = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
        except Exception as error:
            raise RuntimeError(f"Core ML failed to load {package}; no PyTorch fallback: {error}") from error
        self.models[board_size] = compiled
        return compiled

    def predict(self, features):
        features = np.asarray(features, dtype=np.float32)
        if (features.ndim != 4 or features.shape[0] < 1 or
                features.shape[1] != self.model.config.channels or
                features.shape[2] != features.shape[3] or not 2 <= features.shape[2] <= 19):
            raise ValueError("Core ML inference expects nonempty [batch,21,N,N], N in [2,19]")
        model = self._load_model(features.shape[2])
        parts = []
        for start in range(0, len(features), self.batch_size):
            chunk = features[start:start + self.batch_size]
            padded = np.zeros((self.batch_size, *features.shape[1:]), dtype=np.float32)
            padded[:len(chunk)] = chunk
            try:
                result = model.predict({"features": padded})
            except Exception as error:
                raise RuntimeError(f"Core ML CPU_AND_NE prediction failed for {features.shape[2]}x{features.shape[2]} "
                                   f"(batch {self.batch_size}): {error}") from error
            outputs = {name: np.asarray(result[name], dtype=np.float32)[:len(chunk)].copy()
                       for name in OUTPUT_NAMES}
            if any(not np.isfinite(value).all() for value in outputs.values()):
                raise RuntimeError("Core ML produced non-finite predictions")
            parts.append(outputs)
        return {name: np.concatenate([part[name] for part in parts], axis=0) for name in OUTPUT_NAMES}

    def compute_plan_report(self, board_size):
        """Static device preferences, never a claim about runtime utilization."""
        try:
            model = self._load_model(board_size)
            plan = self.ct.models.compute_plan.MLComputePlan.load_from_path(
                model.get_compiled_model_path(), compute_units=self.ct.ComputeUnit.CPU_AND_NE)
            counts = Counter()
            labels = {"MLNeuralEngineComputeDevice": "neural_engine", "MLCPUComputeDevice": "cpu",
                      "MLGPUComputeDevice": "gpu"}

            def visit(block):
                for operation in block.operations:
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                    name = type(usage.preferred_compute_device).__name__ if usage else "unknown"
                    counts[labels.get(name, "unclassified")] += 1
                    for nested in operation.blocks:
                        visit(nested)

            for function in plan.model_structure.program.functions.values():
                visit(function.block)
            return {"available": True, "evidence": "static_compute_plan",
                    "preferred_operations": dict(counts), "runtime_utilization_measured": False}
        except Exception as error:
            return {"available": False, "reason": str(error), "runtime_utilization_measured": False}

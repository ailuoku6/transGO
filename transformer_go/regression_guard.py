"""Anchor-based regression guard.

After each fixed-anchor match the guard compares the anchor win rate with the
best one seen so far for the same anchor. A new high stores a restore point
(best and learner weights). A drop of at least ``rollback_drop`` restores both
best.pt and learner.pt from that point. After MAX_ROLLBACKS_PER_RESTORE
rollbacks to one point, a further drop rebases the reference on the current
result instead, because a high that cannot be reproduced was probably luck.
Every decision is returned as a JSON record; the guard state lives in
OUTPUT/anchors/regression-guard.json.
"""

import json
from pathlib import Path
import shutil

import torch

from .checkpoint import model_version, require_selfplay_origin

GUARD_PROTOCOL = "anchor-rollback-v1"
STATE_NAME = "regression-guard.json"
# A lucky high could otherwise trigger rollbacks forever.
MAX_ROLLBACKS_PER_RESTORE = 2
EPSILON = 1e-9


def _atomic_write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _atomic_copy(source, target):
    target = Path(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _checkpoint_step(path):
    return int(torch.load(path, map_location="cpu", weights_only=True)["step"])


def _restore_point_intact(point):
    paths = {point["best_path"]: point["best_sha256"], point["learner_path"]: point["learner_sha256"]}
    return all(Path(path).is_file() and model_version(path) == digest for path, digest in paths.items())


def _save_restore_point(directory, iteration, best_path, learner_path, winrate):
    best_sha256, learner_sha256 = model_version(best_path), model_version(learner_path)
    best_copy = directory / f"restore-{iteration:04d}-best.pt"
    _atomic_copy(best_path, best_copy)
    learner_copy = best_copy
    if learner_sha256 != best_sha256:
        learner_copy = directory / f"restore-{iteration:04d}-learner.pt"
        _atomic_copy(learner_path, learner_copy)
    return {"iteration": iteration, "winrate": winrate,
            "best_path": str(best_copy), "best_sha256": best_sha256,
            "learner_path": str(learner_copy), "learner_sha256": learner_sha256,
            "learner_step": _checkpoint_step(learner_copy)}


def _prune_restore_point(point, keep):
    for key in ("best_path", "learner_path"):
        path = Path(point[key])
        if str(path) not in keep and path.name.startswith("restore-"):
            path.unlink(missing_ok=True)


def _rollback(point, best_path, learner_path):
    """Restore weights and optimizer state; keep the step counter and RNG moving forward."""
    if not _restore_point_intact(point):
        raise ValueError("restore point changed on disk")
    current = torch.load(learner_path, map_location="cpu", weights_only=True)
    current_step = int(current["step"])
    payload = torch.load(point["learner_path"], map_location="cpu", weights_only=True)
    require_selfplay_origin(payload)
    # Loss logs and training-loss.csv are keyed by global step, so continue
    # counting from the pre-rollback learner instead of reusing old steps.
    payload["step"] = max(current_step, int(payload["step"]))
    # Reusing the old sampling stream would replay the same minibatch indices.
    payload["numpy_rng"], payload["torch_rng"] = current.get("numpy_rng"), current["torch_rng"]
    learner_path = Path(learner_path)
    temporary = learner_path.with_suffix(learner_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(learner_path)
    _atomic_copy(point["best_path"], best_path)
    return current_step, payload["step"]


def apply_regression_guard(root, iteration, best_path, learner_path, anchor_sha256, winrate,
                           rollback_drop, report=print):
    directory = Path(root) / "anchors"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / STATE_NAME
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    record = {"protocol": GUARD_PROTOCOL, "iteration": iteration, "anchor_sha256": anchor_sha256,
              "winrate": winrate, "rollback_drop": rollback_drop,
              "max_rollbacks_per_restore": MAX_ROLLBACKS_PER_RESTORE}
    state = previous if previous and previous.get("anchor_sha256") == anchor_sha256 else None
    if previous is not None and state is None:
        record["reset_reason"] = "anchor_changed"
        # Win rates against a different anchor are not comparable; drop the old point.
        _prune_restore_point(previous["restore_point"], set())
    point = state["restore_point"] if state else None
    if point is not None and not _restore_point_intact(point):
        record["reset_reason"] = "restore_point_missing"
        point = None
    drop = None if point is None else point["winrate"] - winrate
    limit_reached = (point is not None and drop + EPSILON >= rollback_drop
                     and state["rollbacks_to_restore_point"] >= MAX_ROLLBACKS_PER_RESTORE)
    if point is None or winrate > point["winrate"] + EPSILON or limit_reached:
        new_point = _save_restore_point(directory, iteration, best_path, learner_path, winrate)
        if point is not None:
            _prune_restore_point(point, {new_point["best_path"], new_point["learner_path"]})
        record.update(decision="rebase_after_rollback_limit" if limit_reached else "new_restore_point",
                      restore_point=new_point, replaced_restore_point=point, drop=drop)
        state = {"anchor_sha256": anchor_sha256, "restore_point": new_point,
                 "rollbacks_to_restore_point": 0}
    else:
        count = state["rollbacks_to_restore_point"]
        record.update(restore_point=point, drop=drop, rollbacks_to_restore_point=count)
        if drop + EPSILON < rollback_drop:
            record["decision"] = "hold"
        else:
            before = {"best_sha256": model_version(best_path),
                      "learner_sha256": model_version(learner_path)}
            step_before, step_after = _rollback(point, best_path, learner_path)
            state["rollbacks_to_restore_point"] = count + 1
            record.update(decision="rollback", rollbacks_to_restore_point=count + 1,
                          before={**before, "learner_step": step_before},
                          after={"best_sha256": model_version(best_path),
                                 "learner_sha256": model_version(learner_path),
                                 "learner_step": step_after,
                                 "restored_learner_step": point["learner_step"]})
    state["last_check"] = {"iteration": iteration, "winrate": winrate, "decision": record["decision"]}
    _atomic_write_json(state_path, state)
    report({"event": "regression_guard", **{key: record[key] for key in
                                            ("iteration", "winrate", "decision")},
            "reference_winrate": record["restore_point"]["winrate"]})
    return record

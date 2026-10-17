import numpy as np
import pytest
import torch
from transformer_go._native import Board
from transformer_go.checkpoint import load_checkpoint, save_checkpoint
from transformer_go.config import load_config
from transformer_go.data import write_shard
from transformer_go.model import PolicyValueNet
from transformer_go.pipeline import run_pipeline
from transformer_go.runtime import seed_all
from transformer_go.train import train


@pytest.mark.parametrize("source", ["sgf", "expert", None])
def test_training_rejects_human_or_unknown_source_before_update(tmp_path, source):
    config = load_config("configs/smoke.yaml")
    board = Board(3, 0.5)
    data = {"features": board.features()[None], "legal": np.ones((1, 10), dtype=bool),
            "policy": np.full((1, 10), 0.1, dtype=np.float32), "value": np.array([0]),
            "score": np.zeros(1, dtype=np.float32), "ownership": np.ones((1, 3, 3), dtype=np.int64),
            "moves_left": np.ones(1, dtype=np.float32), "aux_mask": np.ones(1, dtype=np.float32)}
    write_shard(tmp_path/"replay", data, {"source": source})
    model = PolicyValueNet(config.model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="selfplay-only"):
        train(config, model, torch.device("cpu"), tmp_path/"replay", tmp_path/"model.pt", 1)
    assert not (tmp_path/"model.pt").exists()
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@pytest.mark.parametrize("origin", ["human_pretrained", None])
def test_training_checkpoint_rejects_unverified_origin(tmp_path, origin):
    config = load_config("configs/smoke.yaml")
    path = tmp_path/"legacy.pt"
    model = PolicyValueNet(config.model)
    save_checkpoint(path, model, config)
    payload = torch.load(path, weights_only=True)
    if origin is None: payload.pop("training_origin")
    else: payload["training_origin"] = origin
    torch.save(payload, path)
    with pytest.raises(ValueError, match="selfplay origin"):
        load_checkpoint(path, torch.device("cpu"), selfplay_only=True)
    # Old artifacts can still be inspected or used for GTP without joining Zero training.
    load_checkpoint(path, torch.device("cpu"))
    with pytest.raises(ValueError, match="selfplay origin"):
        train(config, model, torch.device("cpu"), tmp_path/"replay", tmp_path/"out.pt", 1, payload)


def test_zero_pipeline_random_init_two_rounds(tmp_path):
    from transformer_go.data import read_shard
    config = load_config("configs/smoke.yaml")
    seed_all(config.seed)
    run_pipeline(config, torch.device("cpu"), tmp_path, iterations=2, selfplay_games=1,
                 train_steps=1, eval_games=2, report=lambda _: None)
    for name in ("best.pt", "learner.pt", "candidate-0001.pt", "candidate-0002.pt"):
        _, _, payload = load_checkpoint(tmp_path/name, torch.device("cpu"), selfplay_only=True)
        assert payload["training_origin"] == "random_init_selfplay"
        if name == "learner.pt": assert payload["step"] == 2
        if name == "best.pt": assert payload["step"] == 0
    paths = list((tmp_path/"replay").glob("*.npz"))
    assert len(paths) == 2
    for path in paths: assert read_shard(path)[1]["source"] == "selfplay"

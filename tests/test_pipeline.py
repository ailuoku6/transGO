import io
import numpy as np
import pytest
import torch
from transformer_go.config import load_config
from transformer_go.runtime import seed_all
from transformer_go.model import PolicyValueNet
from transformer_go.search import Evaluator
from transformer_go.selfplay import generate
from transformer_go.data import read_shard, augment
from transformer_go.train import train, losses
from transformer_go.checkpoint import load_checkpoint
from transformer_go.sgf import parse
from transformer_go._native import Board
from transformer_go.gtp import GTP, to_action, to_coord


def test_selfplay_train_resume_sgf_roundtrip(tmp_path):
    config = load_config("configs/smoke.yaml"); config.device = "cpu"
    seed_all(config.seed)
    model = PolicyValueNet(config.model)
    paths = generate(config, Evaluator(model, torch.device("cpu")), tmp_path/"replay", 2, report=lambda _: None)
    data, meta = read_shard(paths[0])
    assert meta["moves"] == len(data["value"])
    nodes = parse(paths[0].with_suffix(".sgf").read_text())
    board = Board(config.board_size, config.komi)
    players = []
    for i, node in enumerate(nodes[1:]):
        np.testing.assert_allclose(data["features"][i], board.features())
        players.append(board.to_play)
        coordinate = node["B" if board.to_play == 1 else "W"][0]
        action = config.board_size**2 if not coordinate else (ord(coordinate[1])-97)*config.board_size+ord(coordinate[0])-97
        board = board.play(action)
    assert board.terminal()
    np.testing.assert_array_equal(data["value"], [int(board.result()*p)+1 for p in players])
    np.testing.assert_allclose(data["score"], [board.score()*p/config.board_size**2 for p in players])
    output = tmp_path/"model.pt"
    before = model.stem.weight.detach().clone()
    metrics = train(config, model, torch.device("cpu"), tmp_path/"replay", output, 2, report=lambda _: None)
    assert np.isfinite(metrics["total"]) and not torch.equal(before, model.stem.weight)
    restored, saved_config, payload = load_checkpoint(output, torch.device("cpu"))
    assert payload["step"] == 2 and saved_config.to_dict() == config.to_dict()
    assert payload["training_origin"] == "random_init_selfplay"
    metrics = train(config, restored, torch.device("cpu"), tmp_path/"replay", output, 1, payload, lambda _: None)
    assert metrics["step"] == 3
    seed_all(config.seed)
    continuous = PolicyValueNet(config.model)
    train(config, continuous, torch.device("cpu"), tmp_path/"replay", tmp_path/"continuous.pt", 3, report=lambda _: None)
    for key, value in continuous.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_symmetry_keeps_legal_policy_aligned():
    n = 3
    data = {"features": np.arange(21*n*n, dtype=np.float32).reshape(1, 21, n, n),
            "legal": np.zeros((1, n*n+1), dtype=bool), "policy": np.zeros((1, n*n+1), dtype=np.float32),
            "ownership": np.arange(n*n).reshape(1, n, n), "value": np.array([2]),
            "score": np.array([1.0]), "moves_left": np.array([1.0]), "aux_mask": np.array([1.0])}
    data["legal"][0, 1] = True; data["policy"][0, 1] = 1
    for seed in range(16):
        result = augment(data, np.random.default_rng(seed))
        assert np.array_equal(result["legal"], result["policy"].astype(bool))
        assert result["policy"][0, -1] == 0


def test_gtp_protocol_ids_errors_and_19x19():
    for a in range(362): assert to_action(to_coord(a, 19), 19) == a
    with pytest.raises(ValueError): to_action("I9", 19)
    with pytest.raises(ValueError): to_action("A20", 19)
    engine = GTP(load_config("configs/smoke.yaml"), None)
    output = io.StringIO()
    engine.run(io.StringIO("1 protocol_version\n2 play b A1\n3 play w A1\n4 nonsense\n5 undo\n6 quit\n"), output)
    text = output.getvalue()
    assert "=1 2\n\n" in text and "?3 illegal move" in text and "?4 unknown command" in text
    assert engine.board.move_number == 0


def test_sgf_parser_strict_and_escaped():
    assert parse(r"(;SZ[9]C[a\]b (comment)]RE[B+R];B[aa])")[0]["C"] == ["a]b (comment)"]
    with pytest.raises(ValueError): parse("(;SZ[9];B[aa](;W[bb])(;W[cc]))")


def test_masked_policy_loss_has_finite_gradients():
    config = load_config("configs/smoke.yaml")
    model = PolicyValueNet(config.model)
    legal = torch.ones(2, 10, dtype=torch.bool); legal[:, 0] = False
    batch = {"legal": legal, "policy": legal.float()/9, "value": torch.tensor([0, 2]),
             "score": torch.zeros(2), "ownership": torch.ones(2, 3, 3, dtype=torch.long),
             "moves_left": torch.ones(2), "aux_mask": torch.ones(2)}
    result = losses(model(torch.zeros(2, 21, 3, 3), legal), batch, config.train)
    result["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

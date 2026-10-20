import numpy as np
import pytest

from transformer_go._native import Board, Search, corrected_selection_weights, shaped_dirichlet_alphas
from transformer_go.config import load_config
from transformer_go.exploration import selfplay_temperature, exploration_settings
from transformer_go.search import choose_action
import transformer_go.selfplay as selfplay


def uniform(boards):
    return (np.ones((len(boards), boards[0].size ** 2 + 1), np.float32),
            np.zeros(len(boards), np.float32))


def test_temperature_scales_with_board_and_legacy_is_unchanged():
    for size in (3, 9, 19):
        assert selfplay_temperature(0, size, "katago", 20) == .75
        assert selfplay_temperature(size, size, "katago", 20) == pytest.approx(.45)
        assert selfplay_temperature(size * 100, size, "katago", 20) == pytest.approx(.15)
    assert selfplay_temperature(19, 9, "legacy", 20) == 1
    assert selfplay_temperature(20, 9, "legacy", 20) == .1
    config = load_config("configs/tiny.yaml")
    before = config.to_dict()
    assert exploration_settings(config, "katago")["root_noise_total_concentration"] == 10.83
    assert config.to_dict() == before


def test_shaped_noise_respects_legality_floor_and_uniform_fallback():
    priors = [0.9, 0.0999, 0.0001, 0]
    alphas = np.asarray(shaped_dirichlet_alphas(priors, [1, 0, 1, 1]))
    assert alphas[1] == 0
    assert alphas.sum() == pytest.approx(10.83)
    assert np.all(alphas[[0, 2, 3]] >= 10.83 / 6)
    assert alphas[0] > alphas[2] >= alphas[3]
    np.testing.assert_allclose(shaped_dirichlet_alphas([0, 0, 0], [1, 1, 0]), [5.415, 5.415, 0])
    with pytest.raises(ValueError):
        shaped_dirichlet_alphas([1], [0])


def test_correction_prunes_bad_exploration_but_keeps_rare_good_move():
    # A one-visit move with high Q must survive. Only the bad one is removed.
    visits = [100, 1, 1, 0]
    weights = corrected_selection_weights(visits, [.5, -.5, .9, 0], [.8, .001, .001, .198], 15)
    assert weights == [100, 0, 1, 0]
    assert visits == [100, 1, 1, 0]
    # At tiny budgets even ties must remain valid; no arbitrary 1% cutoff.
    assert corrected_selection_weights([1, 1], [0, 0], [.5, .5], 2) == [1, 1]
    with pytest.raises(ValueError):
        corrected_selection_weights([0, 0], [0, 0], [.5, .5], 2)


def test_target_correction_does_not_change_search_or_reused_tree():
    board, tree = Board(5), Search(seed=3)
    raw = tree.run(board, uniform, 80, 4, reuse=True)
    corrected = tree.run(board, uniform, 80, 4, reuse=True, policy_correction=True)
    assert corrected["new_simulations"] == 0
    assert corrected["visits"] == raw["visits"]
    assert corrected["policy"] == raw["policy"]
    assert corrected["value"] == raw["value"]
    assert sum(corrected["selection_policy"]) == pytest.approx(1)
    assert np.all(np.asarray(corrected["selection_weights"]) <= raw["visits"])
    action = int(np.argmax(raw["visits"]))
    assert tree.advance(action)
    next_result = tree.run(board.play(action), uniform, 80, 4, reuse=True, policy_correction=True)
    assert next_result["reused_simulations"] == raw["visits"][action] - 1
    assert sum(next_result["visits"]) == 80
    assert next_result["selection_policy"][action] == 0


def test_shaped_noise_replaces_old_root_noise_with_same_seed():
    board, a, b = Board(5), Search(seed=1), Search(seed=1)
    a.run(board, uniform, 64, 4, reuse=True)
    b.run(board, uniform, 64, 4, reuse=True)
    a.run(board, uniform, 64, 4, epsilon=.75, shaped_noise=True, reuse=True)
    a.reseed(23)
    b.reseed(23)
    assert a.run(board, uniform, 80, 4, epsilon=.25, shaped_noise=True, reuse=True) == b.run(
        board, uniform, 80, 4, epsilon=.25, shaped_noise=True, reuse=True)


@pytest.mark.parametrize("profile", ["katago", "legacy"])
def test_selfplay_saves_soft_corrected_target_not_sampled_move(monkeypatch, profile):
    calls = []
    targets = []

    def fake_search(board, evaluator, config, **kwargs):
        calls.append(kwargs)
        raw = np.zeros(board.size ** 2 + 1)
        raw[0], raw[-1] = .5, .5
        selection = raw.copy()
        if kwargs["policy_correction"]:
            selection[0], selection[-1] = .2, .8
        return {"policy": raw, "visits": raw * 10, "selection_policy": selection,
                "selection_weights": selection * 5, "simulations": 10}

    def choose_pass(policy, temperature, rng):
        targets.append((policy.copy(), temperature))
        return 9

    monkeypatch.setattr(selfplay, "search", fake_search)
    monkeypatch.setattr(selfplay, "choose_action", choose_pass)
    config = load_config("configs/smoke.yaml")
    data, meta, _ = selfplay.play_game(config, None, 42, selfplay_policy=profile)
    assert meta["moves"] == 2 and meta["exploration"]["profile"] == profile
    expected = [.2, .8] if profile == "katago" else [.5, .5]
    np.testing.assert_allclose(data["policy"][:, [0, 9]], [expected] * 2)
    assert data["policy"][0, 9] < 1  # Actual pass is not a one-hot training label.
    assert all(call["policy_correction"] == (profile == "katago") for call in calls)
    assert targets[0][1] == (.75 if profile == "katago" else 1)


def test_lower_temperature_reduces_low_visit_sampling_mass():
    # The diagnosed 59 singleton moves jointly weigh 29.5% at T=1.
    visits = np.array([1] * 59 + [2] * 5 + [3] * 5 + [4, 5, 7, 7, 8, 9, 9, 9, 27, 31])
    assert visits.sum() == 200
    tau = selfplay_temperature(2, 9, "katago", 20)
    probabilities = visits ** (1 / tau)
    probabilities /= probabilities.sum()
    assert probabilities[:59].sum() == pytest.approx(.10183577)
    # Exercise the real sampler with the fixed distribution and fixed seed.
    rng = np.random.default_rng(27)
    rate = sum(choose_action(visits, tau, rng) < 59 for _ in range(3000)) / 3000
    assert .08 < rate < .13

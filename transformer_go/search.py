import numpy as np
import torch
from ._native import Board, Search
from .config import SearchConfig


class Evaluator:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.calls = 0
        self.positions = 0

    @torch.inference_mode()
    def predict(self, boards):
        self.model.eval()
        features = torch.from_numpy(np.stack([b.features() for b in boards])).to(self.device)
        mask = torch.tensor([b.legal_mask() for b in boards], dtype=torch.bool, device=self.device)
        outputs = self.model(features, mask)
        policies = outputs["policy_logits"].softmax(-1)
        wdl = outputs["value_logits"].softmax(-1)
        values = wdl[:, 2] - wdl[:, 0]
        self.calls += 1
        self.positions += len(boards)
        return outputs, policies, wdl, values

    def __call__(self, boards):
        outputs, policies, wdl, values = self.predict(boards)
        scores = outputs["score_mean"]
        return (policies.cpu().numpy().astype(np.float32),
                values.cpu().numpy().astype(np.float32),
                scores.cpu().numpy().astype(np.float32))


def analyze_position(board: Board, evaluator, config: SearchConfig, seed=0, score_utility=0.0):
    """Return MCTS plus model shape/value judgment for one position."""
    from .judgment import position_report

    outputs, policies, wdl, values = evaluator.predict([board])
    result = search(board, evaluator, config, seed=seed, score_utility=score_utility)
    own = outputs["ownership_logits"].softmax(-1)[0].cpu().numpy().astype(np.float32)
    prediction = {
        "wdl": wdl[0].cpu().numpy().astype(np.float32),
        "score_mean": float(outputs["score_mean"][0].cpu()),
        "score_logvar": float(outputs["score_logvar"][0].exp().cpu()),
        "moves_left": float(outputs["moves_left"][0].cpu()),
        "ownership": own,
    }
    result["model"] = position_report(board, prediction)
    result["model"]["policy"] = policies[0].cpu().numpy().astype(np.float32).tolist()
    return result


def search(board: Board, evaluator, config: SearchConfig, seed=0, noise=False, min_moves=0,
           score_utility=0.0):
    return Search(config.c_puct, config.virtual_loss, seed).run(
        board, evaluator, config.simulations, config.batch_size,
        config.dirichlet_alpha, config.dirichlet_epsilon if noise else 0.0, min_moves, score_utility,
    )


def choose_action(policy, temperature: float, rng: np.random.Generator) -> int:
    policy = np.asarray(policy, dtype=np.float64)
    if temperature <= 0:
        return int(policy.argmax())
    logits = np.full_like(policy, -np.inf)
    positive = policy > 0
    logits[positive] = np.log(policy[positive]) / temperature
    probs = np.exp(logits - logits.max())
    return int(rng.choice(len(policy), p=probs / probs.sum()))

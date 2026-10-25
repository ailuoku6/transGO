import numpy as np
import torch
from ._native import Board, Search
from .config import SearchConfig


class Evaluator:
    def __init__(self, model, device, *, backend="torch", coreml_batch_size=16,
                 coreml_cache_dir="runs/coreml"):
        if backend not in ("torch", "coreml"):
            raise ValueError("inference backend must be torch or coreml")
        self.model, self.device = model, device
        self.backend = backend
        self.coreml = None
        if backend == "coreml":
            from .coreml import CoreMLBackend
            self.coreml = CoreMLBackend(model, coreml_batch_size, coreml_cache_dir)
        self.calls = 0
        self.positions = 0

    def prepare(self, board_size):
        """Compile before starting workers with bounded RPC request timeouts."""
        if self.coreml is not None:
            self.coreml._load_model(board_size)

    @torch.inference_mode()
    def predict(self, boards):
        return self.predict_arrays(np.stack([b.features() for b in boards]),
                                   np.asarray([b.legal_mask() for b in boards], dtype=bool))

    @torch.inference_mode()
    def predict_arrays(self, features, legal):
        if self.coreml is None:
            self.model.eval()
            features = torch.from_numpy(features).to(self.device)
            mask = torch.from_numpy(legal).to(self.device, dtype=torch.bool)
            outputs = self.model(features, mask)
        else:
            outputs = {name: torch.from_numpy(value) for name, value in self.coreml.predict(features).items()}
            mask = torch.as_tensor(legal, dtype=torch.bool)
            outputs["policy_logits"] = outputs["policy_logits"].masked_fill(~mask, float("-inf"))
        policies = outputs["policy_logits"].softmax(-1)
        wdl = outputs["value_logits"].softmax(-1)
        values = wdl[:, 2] - wdl[:, 0]
        self.calls += 1
        self.positions += len(features)
        return outputs, policies, wdl, values

    def __call__(self, boards):
        return self.evaluate_arrays(np.stack([b.features() for b in boards]),
                                    np.asarray([b.legal_mask() for b in boards], dtype=bool))

    def evaluate_arrays(self, features, legal):
        outputs, policies, wdl, values = self.predict_arrays(features, legal)
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


class SearchSession:
    """One game and one model's tree; advance after every actual move."""
    def __init__(self):
        self.tree = None
        self.parameters = None

    def advance(self, action):
        return self.tree.advance(action) if self.tree is not None else False

    def reset(self):
        if self.tree is not None:
            self.tree.reset()

    def prepare(self, config, seed):
        parameters = (config.c_puct, config.virtual_loss)
        if self.tree is None or parameters != self.parameters:
            self.tree = Search(*parameters, seed)
            self.parameters = parameters
        self.tree.reseed(seed)
        return self.tree


def search(board: Board, evaluator, config: SearchConfig, seed=0, noise=False, min_moves=0,
           score_utility=0.0, session=None, *, shaped_noise=False, policy_correction=False):
    tree = (Search(config.c_puct, config.virtual_loss, seed) if session is None
            else session.prepare(config, seed))
    return tree.run(
        board, evaluator, config.simulations, config.batch_size,
        config.dirichlet_alpha, config.dirichlet_epsilon if noise else 0.0, min_moves, score_utility,
        reuse=session is not None, shaped_noise=shaped_noise, policy_correction=policy_correction,
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

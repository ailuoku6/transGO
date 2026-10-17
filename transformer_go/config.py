from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    channels: int = 21
    width: int = 192
    layers: int = 8
    heads: int = 6
    window: int = 5
    global_every: int = 2

    def __post_init__(self):
        if self.width < 8 or self.heads < 1 or self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.layers < 1 or self.window < 1 or self.window % 2 == 0 or self.global_every < 1:
            raise ValueError("invalid Transformer configuration")
        if self.channels != 21:
            raise ValueError("feature schema v1 requires 21 channels")


@dataclass
class SearchConfig:
    simulations: int = 200
    batch_size: int = 16
    c_puct: float = 1.5
    virtual_loss: float = 1.0
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 20


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 0.0002
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    replay_positions: int = 2000000
    score_weight: float = 0.3
    ownership_weight: float = 0.2
    moves_weight: float = 0.1


@dataclass
class Config:
    board_size: int = 9
    komi: float = 7.5
    # KataGo-compatible rule name understood by the native rules engine.
    # Keep the historical default so existing checkpoints and replay shards
    # continue to load without migration.
    rules: str = "chinese-area-psk"
    device: str = "auto"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if not 2 <= self.board_size <= 19:
            raise ValueError("board_size must be in [2,19]")
        if not isinstance(self.rules, str) or not self.rules.strip():
            raise ValueError("rules must be a non-empty string")

    def to_dict(self):
        return asdict(self)


def from_dict(raw: dict) -> Config:
    raw = dict(raw)
    for key, cls in (("model", ModelConfig), ("search", SearchConfig), ("train", TrainConfig)):
        raw[key] = cls(**raw.get(key, {}))
    return Config(**raw)


def load_config(path: str | Path) -> Config:
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})

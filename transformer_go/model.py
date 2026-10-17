import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * 8)
        self.down = nn.Linear(width * 4, width)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attention(h, h, h, attn_mask=mask, need_weights=False)[0]
        a, b = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(a) * b)


class PolicyValueNet(nn.Module):
    """Dynamic board size; local attention masks alternate with full attention.

    Local masks express locality but use dense kernels in v1 (not sparse windows).
    All MPS operations use float32 to avoid backend-specific AMP/float64 failures.
    Value class order: loss, draw, win, from the player to move's perspective.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.width
        self.stem = nn.Conv2d(config.channels, d, 3, padding=1)
        self.position = nn.Linear(6, d, bias=False)
        self.pass_token = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(Block(d, config.heads) for _ in range(config.layers))
        self.norm = nn.LayerNorm(d)
        self.policy = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
        self.score = nn.Linear(d, 2)
        self.ownership = nn.Linear(d, 3)
        self.moves = nn.Linear(d, 1)

    def forward(self, features, legal_mask=None):
        b, _, n, m = features.shape
        if n != m or features.shape[1] != self.config.channels:
            raise ValueError("expected [batch,21,N,N]")
        axis = torch.linspace(-1, 1, n, device=features.device, dtype=features.dtype)
        row, col = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack([row, col, (math.pi*row).sin(), (math.pi*row).cos(),
                              (math.pi*col).sin(), (math.pi*col).cos()], dim=-1).reshape(n*n, 6)
        x = self.stem(features).flatten(2).transpose(1, 2) + self.position(coords)
        # Pass token gets board context on every global-attention block.
        x = torch.cat([x, self.pass_token.expand(b, -1, -1)], dim=1)
        ij = torch.arange(n, device=features.device)
        rr, cc = torch.meshgrid(ij, ij, indexing="ij")
        points = torch.stack([rr.flatten(), cc.flatten()], dim=1)
        distant = (points[:, None] - points[None, :]).abs().amax(-1) > self.config.window // 2
        local_mask = torch.zeros(n*n+1, n*n+1, dtype=torch.bool, device=features.device)
        local_mask[:-1, :-1] = distant
        # Prevent pass from becoming a global shortcut on local-only blocks.
        local_mask[-1, :-1] = True
        local_mask[:-1, -1] = True
        for i, block in enumerate(self.blocks):
            x = block(x, None if (i+1) % self.config.global_every == 0 else local_mask)
        x = self.norm(x)
        pooled = x.mean(1)
        logits = self.policy(x).squeeze(-1)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), float("-inf"))
        score = self.score(pooled)
        return {"policy_logits": logits, "value_logits": self.value(pooled),
                "score_mean": score[:, 0], "score_logvar": score[:, 1].clamp(-8, 8),
                "ownership_logits": self.ownership(x[:, :-1]).reshape(b, n, n, 3),
                "moves_left": F.softplus(self.moves(pooled).squeeze(-1))}

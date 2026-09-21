"""Compact dual-encoder VLM: CNN visual tower + embedding question tower.

Width is parameterized by `scale`, so teacher and student share the exact
architecture family and only differ in capacity — the cleanest possible
setup for studying how much distillation buys back per parameter spent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    scale: float = 1.0        # width multiplier vs. the reference teacher
    hidden: int = 64
    max_len: int = 6          # padded question length
    embed_dim: int = 24


class VisualTower(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        c1, c2 = max(4, out_dim // 4), max(8, out_dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(c2, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(images))


class QuestionTower(nn.Module):
    def __init__(self, vocab: int, spec: ModelSpec, out_dim: int):
        super().__init__()
        e = max(8, int(spec.embed_dim * spec.scale))
        self.embed = nn.Embedding(vocab, e, padding_idx=0)
        self.proj = nn.Linear(e, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).float().unsqueeze(-1)
        emb = self.embed(tokens) * mask
        pooled = emb.sum(1) / mask.sum(1).clamp(min=1.0)
        return self.proj(pooled)


class MiniVLM(nn.Module):
    def __init__(self, vocab: int, n_answers: int, spec: ModelSpec | None = None):
        super().__init__()
        spec = spec or ModelSpec()
        h = max(16, int(spec.hidden * spec.scale))
        self.spec = spec
        self.vision = VisualTower(h)
        self.question = QuestionTower(vocab, spec, h)
        self.head = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, n_answers)
        )

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        v = self.vision(images)
        q = self.question(tokens)
        return self.head(torch.cat([v, q], dim=-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

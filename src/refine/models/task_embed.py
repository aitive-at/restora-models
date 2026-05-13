"""Task embedding + MLP, used to condition NAFBlocks (FiLM) and the
bottleneck transformer (AdaLN)."""
from __future__ import annotations

import torch
from torch import nn


class TaskEmbed(nn.Module):
    def __init__(self, *, num_tasks: int, dim: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_tasks, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(task))

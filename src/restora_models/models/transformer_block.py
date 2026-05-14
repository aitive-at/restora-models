"""Transformer block (MHSA + FFN) with AdaLN conditioning on a task vector.
Adapted from DiT (Peebles & Xie, 2023)."""
from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, *, c: int, task_dim: int, num_heads: int = 8, ffn_dim: int = 256) -> None:
        super().__init__()
        self.adaln1 = nn.Linear(task_dim, 2 * c)
        self.adaln2 = nn.Linear(task_dim, 2 * c)
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, c),
        )

    def forward(self, x: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        gamma1, beta1 = self.adaln1(task_vec).chunk(2, dim=-1)
        gamma2, beta2 = self.adaln2(task_vec).chunk(2, dim=-1)
        gamma1 = gamma1.unsqueeze(1); beta1 = beta1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1); beta2 = beta2.unsqueeze(1)
        h_mod = self.norm1(seq) * (1.0 + gamma1) + beta1
        attn_out, _ = self.attn(h_mod, h_mod, h_mod, need_weights=False)
        seq = seq + attn_out
        h_mod = self.norm2(seq) * (1.0 + gamma2) + beta2
        seq = seq + self.ffn(h_mod)
        return seq.transpose(1, 2).reshape(b, c, h, w)

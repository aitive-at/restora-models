"""Restormer transformer block: MDTA + GDFN, with AdaLN modulation.

References:
- Restormer (Zamir et al., CVPR 2022) — MDTA + GDFN.
- DiT (Peebles & Xie, 2023) — AdaLN-style scalar modulation from a
  conditioning vector. We reuse this for our 5-axis config conditioning
  so a single ConfigEmbed feeds every block.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNormChan(nn.Module):
    """LayerNorm over the channel axis for (B, C, H, W) tensors."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-5)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class MDTA(nn.Module):
    """Multi-Dconv head Transposed Attention. Attention is computed along
    the channel axis (each head sees C/h channels), so cost is O((C/h)^2 * HW)
    instead of vanilla self-attention's O((HW)^2)."""

    def __init__(self, c: int, num_heads: int) -> None:
        super().__init__()
        assert c % num_heads == 0, f"channels {c} not divisible by heads {num_heads}"
        self.num_heads = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, kernel_size=3, padding=1,
                                groups=c * 3, bias=False)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temp
        # FP32 softmax for numerical stability under bf16/fp16 training
        attn = attn.float().softmax(dim=-1).to(v.dtype)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward."""

    def __init__(self, c: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(round(c * expansion))
        self.proj_in = nn.Conv2d(c, hidden * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                            groups=hidden * 2, bias=False)
        self.proj_out = nn.Conv2d(hidden, c, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.dw(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(a) * b)


class RestormerBlock(nn.Module):
    def __init__(self, *, c: int, num_heads: int, task_dim: int,
                 ffn_expansion: float = 2.66) -> None:
        super().__init__()
        self.norm1 = LayerNormChan(c)
        self.attn  = MDTA(c, num_heads=num_heads)
        self.norm2 = LayerNormChan(c)
        self.ffn   = GDFN(c, expansion=ffn_expansion)
        self.adaln1 = nn.Linear(task_dim, 2 * c)
        self.adaln2 = nn.Linear(task_dim, 2 * c)
        # Zero-init AdaLN projections so block starts at identity modulation.
        nn.init.zeros_(self.adaln1.weight); nn.init.zeros_(self.adaln1.bias)
        nn.init.zeros_(self.adaln2.weight); nn.init.zeros_(self.adaln2.bias)

    def _mod(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(b, c, 1, 1); beta = beta.view(b, c, 1, 1)
        return x * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self._mod(self.norm1(x), self.adaln1(task_vec)))
        x = x + self.ffn (self._mod(self.norm2(x), self.adaln2(task_vec)))
        return x

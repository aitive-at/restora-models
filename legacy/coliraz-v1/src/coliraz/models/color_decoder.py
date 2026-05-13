"""Transformer color decoder with learnable color queries (SDPA-backed)."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinePositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats: int = 32, temperature: int = 10000) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_embed = torch.arange(1, h + 1, device=x.device, dtype=x.dtype)[None, :, None].repeat(b, 1, w)
        x_embed = torch.arange(1, w + 1, device=x.device, dtype=x.dtype)[None, None, :].repeat(b, h, 1)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        mem: torch.Tensor,
        query_pos: torch.Tensor,
        mem_pos: torch.Tensor,
    ) -> torch.Tensor:
        q_pe = query + query_pos
        k_pe = mem + mem_pos
        out, _ = self.cross_attn(q_pe, k_pe, mem, need_weights=False)
        query = self.norm1(query + out)
        q_pe = query + query_pos
        out, _ = self.self_attn(q_pe, q_pe, query, need_weights=False)
        query = self.norm2(query + out)
        out = self.ffn(query)
        return self.norm3(query + out)


class _MLP(nn.Module):
    def __init__(self, in_d: int, hid: int, out_d: int, n: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_d
        for _ in range(n - 1):
            layers += [nn.Linear(prev, hid), nn.ReLU(inplace=True)]
            prev = hid
        layers.append(nn.Linear(prev, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiScaleColorDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: list[int],
        num_queries: int = 100,
        hidden_dim: int = 256,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        num_scales: int = 3,
        color_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        assert len(in_channels) == num_scales
        self.num_queries = num_queries
        self.num_scales = num_scales
        self.dec_layers = dec_layers

        self.pe = SinePositionalEncoding(num_pos_feats=hidden_dim // 2)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_scales, hidden_dim)

        self.input_proj = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_channels]
        )
        for p in self.input_proj:
            nn.init.kaiming_uniform_(p.weight, a=1)
            if p.bias is not None:
                nn.init.constant_(p.bias, 0)

        self.layers = nn.ModuleList(
            [
                _TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward)
                for _ in range(dec_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.color_embed = _MLP(hidden_dim, hidden_dim, color_embed_dim, n=3)

    def forward(self, memories: list[torch.Tensor], hi_feat: torch.Tensor) -> torch.Tensor:
        b = hi_feat.shape[0]
        src_seq: list[torch.Tensor] = []
        pos_seq: list[torch.Tensor] = []
        for i, m in enumerate(memories):
            proj = self.input_proj[i](m)
            pos = self.pe(proj)
            level = self.level_embed.weight[i].view(1, -1, 1, 1)
            s = (proj + level).flatten(2).transpose(1, 2)  # (B, HW, C)
            p = pos.flatten(2).transpose(1, 2)
            src_seq.append(s)
            pos_seq.append(p)

        query = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)
        q_pe = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        for i, layer in enumerate(self.layers):
            level = i % self.num_scales
            query = layer(query, src_seq[level], q_pe, pos_seq[level])

        query = self.norm(query)
        emb = self.color_embed(query)  # (B, Q, color_embed_dim)
        return torch.einsum("bqc,bchw->bqhw", emb, hi_feat)

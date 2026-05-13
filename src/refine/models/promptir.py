"""PromptIR backbone — config-driven prompt variant.

4-level Restormer-style U-Net with config-driven PromptBlocks
interleaved on the decoder path. Same forward contract as NAFNet:
forward(rgb: (B,3,H,W) in [0,1], config: (B,5) float) -> (B,3,H,W).
"""
from __future__ import annotations

import torch
from torch import nn

from refine.config import ModelConfig
from .prompt_block import PromptBlock
from .registry import register_model
from .restormer_block import RestormerBlock
from .task_embed import ConfigEmbed


_SIZE_PRESETS: dict[str, dict] = {
    "tiny": {
        "dim": 24, "depths": [2, 2, 2, 2], "refinement": 2,
        "heads": [1, 2, 4, 8],
        "prompt_n": 5, "prompt_dim": 32, "prompt_hw": 8,
    },
    "large": {
        "dim": 48, "depths": [4, 6, 6, 8], "refinement": 4,
        "heads": [1, 2, 4, 8],
        "prompt_n": 5, "prompt_dim": 64, "prompt_hw": 16,
    },
}


def _resolve(cfg: ModelConfig) -> dict:
    preset = _SIZE_PRESETS[cfg.size]
    return {
        "dim":         preset["dim"],
        "depths":      preset["depths"],
        "refinement":  preset["refinement"],
        "heads":       preset["heads"],
        "task_dim":    cfg.task_embed_dim,
        "prompt_n":    cfg.prompt_n   if cfg.prompt_n   is not None else preset["prompt_n"],
        "prompt_dim":  cfg.prompt_dim if cfg.prompt_dim is not None else preset["prompt_dim"],
        "prompt_hw":   cfg.prompt_hw  if cfg.prompt_hw  is not None else preset["prompt_hw"],
    }


def _stack(c: int, n: int, num_heads: int, task_dim: int) -> nn.ModuleList:
    return nn.ModuleList(
        [RestormerBlock(c=c, num_heads=num_heads, task_dim=task_dim) for _ in range(n)]
    )


@register_model("promptir")
class PromptIR(nn.Module):
    def __init__(self, cfg: ModelConfig, *, num_axes: int = 5) -> None:
        super().__init__()
        p = _resolve(cfg)
        dim  = p["dim"]; depths = p["depths"]; heads = p["heads"]
        ref_n = p["refinement"]; task_dim = p["task_dim"]
        assert len(depths) == 4
        assert len(heads) == 4

        self.task_embed = ConfigEmbed(num_axes=num_axes, dim=task_dim)

        self.stem = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        # Encoder
        self.enc_l1 = _stack(dim,       depths[0], heads[0], task_dim)
        self.down1  = nn.PixelUnshuffle(2); self.down1_proj = nn.Conv2d(dim * 4,     dim * 2, 1)
        self.enc_l2 = _stack(dim * 2,   depths[1], heads[1], task_dim)
        self.down2  = nn.PixelUnshuffle(2); self.down2_proj = nn.Conv2d(dim * 2 * 4, dim * 4, 1)
        self.enc_l3 = _stack(dim * 4,   depths[2], heads[2], task_dim)
        self.down3  = nn.PixelUnshuffle(2); self.down3_proj = nn.Conv2d(dim * 4 * 4, dim * 8, 1)
        self.latent = _stack(dim * 8,   depths[3], heads[3], task_dim)

        # Decoder
        self.prompt_l3 = PromptBlock(
            feat_c=dim * 8, prompt_n=p["prompt_n"],
            prompt_dim=p["prompt_dim"], prompt_hw=p["prompt_hw"], cond_dim=task_dim,
        )
        self.up3       = nn.PixelShuffle(2)
        self.up3_proj  = nn.Conv2d(dim * 2,     dim * 4, 1)
        self.skip3     = nn.Conv2d(dim * 4 * 2, dim * 4, 1)
        self.dec_l3    = _stack(dim * 4,   depths[2], heads[2], task_dim)

        self.prompt_l2 = PromptBlock(
            feat_c=dim * 4, prompt_n=p["prompt_n"],
            prompt_dim=p["prompt_dim"], prompt_hw=p["prompt_hw"], cond_dim=task_dim,
        )
        self.up2       = nn.PixelShuffle(2)
        self.up2_proj  = nn.Conv2d(dim,         dim * 2, 1)
        self.skip2     = nn.Conv2d(dim * 2 * 2, dim * 2, 1)
        self.dec_l2    = _stack(dim * 2,   depths[1], heads[1], task_dim)

        self.prompt_l1 = PromptBlock(
            feat_c=dim * 2, prompt_n=p["prompt_n"],
            prompt_dim=p["prompt_dim"], prompt_hw=p["prompt_hw"], cond_dim=task_dim,
        )
        self.up1       = nn.PixelShuffle(2)
        self.up1_proj  = nn.Conv2d(dim // 2,    dim, 1)
        self.skip1     = nn.Conv2d(dim * 2,     dim, 1)
        self.dec_l1    = _stack(dim,       depths[0], heads[0], task_dim)

        self.refinement = _stack(dim, ref_n, heads[0], task_dim)

        self.head = nn.Conv2d(dim, 3, kernel_size=3, padding=1)
        # Default Kaiming init for the head produces a near-identity residual
        # at step 0 while still carrying the prompt-router signal end-to-end,
        # so config conditioning is observable from the very first forward.
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _run(stack: nn.ModuleList, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        for blk in stack:
            x = blk(x, task)
        return x

    def forward(self, rgb: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        task = self.task_embed(config)

        x = self.stem(rgb)

        e1 = self._run(self.enc_l1, x, task)
        e2 = self._run(self.enc_l2, self.down1_proj(self.down1(e1)), task)
        e3 = self._run(self.enc_l3, self.down2_proj(self.down2(e2)), task)
        lat = self._run(self.latent, self.down3_proj(self.down3(e3)), task)

        d = self.prompt_l3(lat, task)
        d = self.up3_proj(self.up3(d))
        d = self.skip3(torch.cat([d, e3], dim=1))
        d = self._run(self.dec_l3, d, task)

        d = self.prompt_l2(d, task)
        d = self.up2_proj(self.up2(d))
        d = self.skip2(torch.cat([d, e2], dim=1))
        d = self._run(self.dec_l2, d, task)

        d = self.prompt_l1(d, task)
        d = self.up1_proj(self.up1(d))
        d = self.skip1(torch.cat([d, e1], dim=1))
        d = self._run(self.dec_l1, d, task)

        d = self._run(self.refinement, d, task)
        return rgb + self.head(d)

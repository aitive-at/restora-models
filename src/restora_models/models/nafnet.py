"""NAFNet compound model."""
from __future__ import annotations

import torch
from torch import nn

from restora_models.config import ModelConfig
from .color import LabToRgb, RgbToLab
from .nafblock import NAFBlock
from .registry import register_model
from .config_embed import ConfigEmbed
from .transformer_block import TransformerBlock


_SIZE_PRESETS: dict[str, dict] = {
    "tiny": {"nf": 32, "enc_depths": [2, 2, 2, 2], "bottle_blocks": 2, "hidden_dim": 256},
    "large": {"nf": 64, "enc_depths": [2, 2, 4, 8], "bottle_blocks": 4, "hidden_dim": 384},
}


def _resolve(cfg: ModelConfig) -> dict:
    preset = _SIZE_PRESETS[cfg.size]
    return {
        "nf": cfg.nf if cfg.nf is not None else preset["nf"],
        "enc_depths": cfg.enc_depths if cfg.enc_depths is not None else preset["enc_depths"],
        "bottle_blocks": cfg.bottle_blocks if cfg.bottle_blocks is not None else preset["bottle_blocks"],
        "hidden_dim": cfg.hidden_dim if cfg.hidden_dim is not None else preset["hidden_dim"],
        "task_dim": cfg.task_embed_dim,
    }


@register_model("nafnet")
class NAFNetMultiTask(nn.Module):
    def __init__(self, cfg: ModelConfig, *, num_axes: int = 5) -> None:
        super().__init__()
        p = _resolve(cfg)
        nf = p["nf"]; depths = p["enc_depths"]; bottle_n = p["bottle_blocks"]
        hidden = p["hidden_dim"]; task_dim = p["task_dim"]
        assert len(depths) == 4

        self.rgb_to_lab = RgbToLab()
        self.lab_to_rgb = LabToRgb()
        self.task_embed = ConfigEmbed(num_axes=num_axes, dim=task_dim)

        self.stem = nn.Conv2d(3, nf, kernel_size=3, padding=1)

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = nf
        enc_channels: list[int] = []
        for n in depths:
            self.enc_stages.append(nn.ModuleList([NAFBlock(ch, task_dim=task_dim) for _ in range(n)]))
            enc_channels.append(ch)
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        self.bottle_in = nn.Conv2d(ch, hidden, kernel_size=1)
        self.bottle = nn.ModuleList([
            TransformerBlock(c=hidden, task_dim=task_dim, num_heads=8, ffn_dim=hidden * 2)
            for _ in range(bottle_n)
        ])
        self.bottle_out = nn.Conv2d(hidden, ch, kernel_size=1)

        self.ups = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for n, skip_c in zip(reversed(depths), reversed(enc_channels)):
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, skip_c * 4, kernel_size=1),
                nn.PixelShuffle(2),
            ))
            ch = skip_c
            self.skip_proj.append(nn.Conv2d(ch * 2, ch, kernel_size=1))
            self.dec_stages.append(nn.ModuleList([NAFBlock(ch, task_dim=task_dim) for _ in range(n)]))

        # Dual output: Lab delta (3 ch, all tasks) + absolute Lab ab (2 ch, colorize axis).
        # Both zero-inited so initial output ~ input via the global Lab residual,
        # and colorize=1 at step 0 produces gray (model learns to add color from there).
        self.head_lab_delta = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.head_ab_abs   = nn.Conv2d(nf, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head_lab_delta.weight)
        nn.init.zeros_(self.head_lab_delta.bias)
        nn.init.zeros_(self.head_ab_abs.weight)
        nn.init.zeros_(self.head_ab_abs.bias)

    def forward(self, rgb: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        lab_n = self.rgb_to_lab(rgb)
        task_vec = self.task_embed(config)

        x = self.stem(lab_n)
        skips: list[torch.Tensor] = []
        for stage, down in zip(self.enc_stages, self.downs):
            for blk in stage:
                x = blk(x, task_vec)
            skips.append(x)
            x = down(x)

        x = self.bottle_in(x)
        for blk in self.bottle:
            x = blk(x, task_vec)
        x = self.bottle_out(x)

        for up, proj, stage, skip in zip(self.ups, self.skip_proj, self.dec_stages, reversed(skips)):
            x = up(x)
            x = proj(torch.cat([x, skip], dim=1))
            for blk in stage:
                x = blk(x, task_vec)

        delta_lab_n = self.head_lab_delta(x)
        ab_pred    = self.head_ab_abs(x)

        # Compose: Lab intermediate carries all-task signal; ab override gated by colorize axis.
        lab_intermediate = lab_n + delta_lab_n
        w   = config[:, 0:1].view(-1, 1, 1, 1)
        ab_out = w * ab_pred + (1.0 - w) * lab_intermediate[:, 1:3]
        L_out  = lab_intermediate[:, 0:1]
        return self.lab_to_rgb(torch.cat([L_out, ab_out], dim=1))

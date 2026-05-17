"""TemporalNAFNet backbone.

Fully convolutional NAFNet-style encoder/decoder with FiLM conditioning
on a 5-axis task vector. Operates on the 28-channel output of
TemporalAlignStem. Bottleneck adds one temporal self-attention block.
Lab dual-head output (Lab-delta for all axes + ab-abs gated by colorize).

All sizes (nano/small/medium/large) registered as separate model types
in the registry but share this class with different hyperparams.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from restora_models.config import ModelConfig
from restora_models.models.config_embed import ConfigEmbed
from restora_models.models.nafblock import NAFBlock
from restora_models.models.registry import register_model
from restora_models.models.temporal_align_stem import TemporalAlignStem
from restora_models.models.transformer_block import TransformerBlock
from restora_models.utils.color import rgb_to_lab, lab_to_rgb


TASK_DIM = 128


@dataclass(frozen=True)
class _SizeSpec:
    nf: int
    enc_depths: tuple[int, int, int, int]
    bottle_blocks: int
    use_temporal_attn: bool


# Widths chosen so nano and small land in the test-spec param bands
# ([4-12M] and [15-30M]) and so every bottleneck channel count is a
# multiple of TransformerBlock's default 8 attention heads.
_SIZES: dict[str, _SizeSpec] = {
    "temporal_nafnet_nano":   _SizeSpec(nf=20, enc_depths=(1, 1, 1, 2), bottle_blocks=2, use_temporal_attn=False),
    "temporal_nafnet_small":  _SizeSpec(nf=28, enc_depths=(2, 2, 2, 4), bottle_blocks=4, use_temporal_attn=True),
    "temporal_nafnet_medium": _SizeSpec(nf=40, enc_depths=(2, 2, 4, 6), bottle_blocks=6, use_temporal_attn=True),
    "temporal_nafnet_large":  _SizeSpec(nf=56, enc_depths=(2, 2, 4, 8), bottle_blocks=8, use_temporal_attn=True),
}


class _DownConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UpConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class _EncoderStage(nn.Module):
    def __init__(self, c: int, depth: int, task_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(c, task_dim=task_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, task)
        return x


class _Bottleneck(nn.Module):
    def __init__(self, c: int, blocks: int, use_attn: bool, task_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(c, task_dim=task_dim) for _ in range(blocks)])
        # CORRECTED: TransformerBlock requires keyword-only c=
        self.attn = TransformerBlock(c=c, task_dim=task_dim) if use_attn else None

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            x = blk(x, task)
            if self.attn is not None and i == len(self.blocks) // 2:
                x = self.attn(x, task)
        return x


class _LabDualHead(nn.Module):
    """Outputs Lab-delta (all axes) + ab-abs (colorize-gated)."""

    def __init__(self, c: int):
        super().__init__()
        self.head_lab_delta = nn.Conv2d(c, 3, 3, padding=1)
        self.head_ab_abs = nn.Conv2d(c, 2, 3, padding=1)
        nn.init.zeros_(self.head_lab_delta.weight)
        nn.init.zeros_(self.head_lab_delta.bias)
        nn.init.zeros_(self.head_ab_abs.weight)
        nn.init.zeros_(self.head_ab_abs.bias)

    def forward(self, feat: torch.Tensor, center_rgb: torch.Tensor, colorize_gate: torch.Tensor) -> torch.Tensor:
        delta = self.head_lab_delta(feat)
        ab_abs = self.head_ab_abs(feat)
        lab = rgb_to_lab(center_rgb)
        lab_new = lab + delta
        gate = colorize_gate.view(-1, 1, 1, 1)
        lab_new = torch.cat([
            lab_new[:, :1],
            lab_new[:, 1:] * (1.0 - gate) + ab_abs * gate,
        ], dim=1)
        return lab_to_rgb(lab_new).clamp(0.0, 1.0)


class TemporalNAFNet(nn.Module):
    def __init__(self, cfg: ModelConfig, num_axes: int = 5):
        super().__init__()
        size = _SIZES[cfg.type]
        nf = size.nf
        self.align_stem = TemporalAlignStem()
        # CORRECTED: ConfigEmbed is keyword-only, no .out_dim - use TASK_DIM constant
        self.cfg_embed = ConfigEmbed(num_axes=num_axes, dim=TASK_DIM)
        task_dim = TASK_DIM
        self.input_conv = nn.Conv2d(28, nf, 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = nf
        for depth in size.enc_depths:
            self.enc.append(_EncoderStage(c, depth, task_dim))
            self.down.append(_DownConv(c, c * 2))
            c = c * 2
        self.bottleneck = _Bottleneck(c, size.bottle_blocks, size.use_temporal_attn, task_dim)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for depth in reversed(size.enc_depths):
            self.up.append(_UpConv(c, c // 2))
            c = c // 2
            self.dec.append(_EncoderStage(c, depth, task_dim))
        self.head = _LabDualHead(c)

    def forward(self, frames: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        task = self.cfg_embed(config)
        center = frames[:, TemporalAlignStem.CENTER_INDEX]
        x = self.align_stem(frames)
        x = self.input_conv(x)
        skips = []
        for stage, down in zip(self.enc, self.down):
            x = stage(x, task)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x, task)
        for up, stage, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = x + skip
            x = stage(x, task)
        return self.head(x, center, config[:, 0])


for _name in _SIZES:
    register_model(_name)(TemporalNAFNet)


from restora_models.models.rsd_refine import RSDRefineHead


_REFINE_WIDTHS = {"nano": 0, "small": 64, "medium": 96, "large": 128}


class TemporalRestora(nn.Module):
    """Backbone + RSD refine in a single module exposing (frames, config) contract.

    For the `nano` size the refine head is skipped (width=0); the model is
    pure backbone for fastest student deployments.
    """

    def __init__(self, cfg: ModelConfig, num_axes: int = 5):
        super().__init__()
        backbone_type = cfg.type.replace("temporal_restora", "temporal_nafnet")
        size_key = backbone_type.rsplit("_", 1)[-1]
        self.backbone = TemporalNAFNet(ModelConfig(type=backbone_type), num_axes=num_axes)
        rw = _REFINE_WIDTHS[size_key]
        self.refine = RSDRefineHead(width=rw, num_axes=num_axes) if rw > 0 else None

    def forward(self, frames: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        coarse = self.backbone(frames, config)
        if self.refine is None:
            return coarse
        return self.refine(coarse, config)


for _size in ("nano", "small", "medium", "large"):
    register_model(f"temporal_restora_{_size}")(TemporalRestora)

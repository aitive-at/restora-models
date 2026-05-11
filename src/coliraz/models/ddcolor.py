"""Top-level DDColor module wiring encoder → pixel decoder → color decoder → refine."""
from __future__ import annotations

import torch
from torch import nn

from coliraz.config import ModelConfig

from .color_decoder import MultiScaleColorDecoder
from .encoder import ConvNeXtEncoder
from .pixel_decoder import PixelDecoder
from .refine import build_refine


class DDColor(nn.Module):
    def __init__(self, cfg: ModelConfig, *, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvNeXtEncoder(
            size=cfg.size, pretrained=pretrained, variant=cfg.encoder_variant
        )
        self.pixel_decoder = PixelDecoder(
            feature_channels=self.encoder.feature_channels, nf=cfg.nf
        )
        hi_ch = cfg.nf // 2
        memory_chs = [cfg.nf, cfg.nf, cfg.nf // 2]
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=memory_chs,
            num_queries=cfg.num_queries,
            hidden_dim=cfg.hidden_dim,
            dec_layers=cfg.dec_layers,
            num_scales=cfg.num_scales,
            color_embed_dim=hi_ch,
        )
        self.refine = build_refine(cfg.num_queries + 3, 2, norm=cfg.refine_norm)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        feats = self.encoder(x_norm)
        memories, hi = self.pixel_decoder(feats)
        coarse = self.color_decoder(memories, hi)
        if coarse.shape[-2:] != x.shape[-2:]:
            coarse = torch.nn.functional.interpolate(
                coarse, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.refine(torch.cat([coarse, x_norm], dim=1))


def build_ddcolor(cfg: ModelConfig, *, pretrained: bool = True) -> DDColor:
    return DDColor(cfg, pretrained=pretrained)

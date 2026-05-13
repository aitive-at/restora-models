"""timm ConvNeXt wrapper exposing multi-scale features."""
from __future__ import annotations

import timm
import torch
from torch import nn

_DEFAULT_VARIANTS = {
    "tiny": "convnext_tiny.fb_in22k",
    "large": "convnext_large.fb_in22k",
}


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        *,
        size: str = "tiny",
        pretrained: bool = True,
        variant: str | None = None,
    ) -> None:
        super().__init__()
        name = variant or _DEFAULT_VARIANTS[size]
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.feature_channels = list(self.backbone.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)

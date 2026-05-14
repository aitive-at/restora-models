"""Loss registry + LossContext + build_loss factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import torch
from torch import nn


@dataclass
class LossContext:
    pred_rgb: torch.Tensor          # (B, 3, H, W)
    clean_rgb: torch.Tensor         # (B, 3, H, W)
    degraded_rgb: torch.Tensor      # (B, 3, H, W)
    config: torch.Tensor            # (B, 5) float — restoration axes
    axes_active: list[str]          # length B; per-sample label like "color+denoise"
    discriminator: nn.Module | None = None


class RestorationLoss(nn.Module):
    name: str = ""

    def forward(self, ctx: LossContext) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


LOSS_REGISTRY: dict[str, Type[RestorationLoss]] = {}


def register_loss(name: str):
    def deco(cls: Type[RestorationLoss]):
        if name in LOSS_REGISTRY:
            raise KeyError(f"loss {name!r} already registered")
        cls.name = name
        LOSS_REGISTRY[name] = cls
        return cls

    return deco


def build_loss(name: str, cfg: dict[str, Any] | None = None) -> RestorationLoss:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"unknown loss {name!r}; have {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**(cfg or {}))

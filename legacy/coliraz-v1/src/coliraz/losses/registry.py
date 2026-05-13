"""Loss registry, LossContext dataclass, and a tiny build_loss factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import torch
from torch import nn


@dataclass
class LossContext:
    pred_ab: torch.Tensor
    gt_ab: torch.Tensor
    pred_rgb: torch.Tensor
    gt_rgb: torch.Tensor
    gray_rgb: torch.Tensor
    discriminator: nn.Module | None = None


class ColorizationLoss(nn.Module):
    """Base class for all coloration losses. Subclasses must set `name`."""
    name: str = ""

    def forward(self, ctx: LossContext) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


LOSS_REGISTRY: dict[str, Type[ColorizationLoss]] = {}


def register_loss(name: str):
    def deco(cls: Type[ColorizationLoss]):
        if name in LOSS_REGISTRY:
            raise KeyError(f"loss {name!r} already registered")
        cls.name = name
        LOSS_REGISTRY[name] = cls
        return cls
    return deco


def build_loss(name: str, cfg: dict[str, Any] | None = None) -> ColorizationLoss:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"unknown loss {name!r}; have {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**(cfg or {}))

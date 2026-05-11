"""Loss aggregation."""
from __future__ import annotations

import torch

from coliraz.config import LossConfig

# Importing the submodules registers the losses
from . import pixel as _pixel  # noqa: F401
from . import perceptual as _perceptual  # noqa: F401
from . import gan as _gan  # noqa: F401
from . import colorfulness as _colorfulness  # noqa: F401
from .gan import GeneratorGANLoss
from .registry import LossContext, build_loss


class LossSet:
    """Composes a list of weighted losses; aggregates totals and emits a flat log dict."""

    def __init__(self, configs: list[LossConfig]):
        self.entries: list[tuple[float, object]] = []
        self.has_gan = False
        self.discriminator_cfg: dict | None = None
        for c in configs:
            loss = build_loss(c.name, c.config)
            self.entries.append((float(c.weight), loss))
            if isinstance(loss, GeneratorGANLoss):
                self.has_gan = True
                self.discriminator_cfg = loss.disc_config

    def parameters(self):
        for _, loss in self.entries:
            for p in loss.parameters():
                yield p

    def to(self, device, dtype=None):
        for _, loss in self.entries:
            loss.to(device, dtype) if dtype is not None else loss.to(device)
        return self

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        total: torch.Tensor | float = 0.0
        log: dict[str, float] = {}
        for weight, loss in self.entries:
            val = loss(ctx)
            total = total + weight * val
            log[loss.name] = float(val.detach())
        if isinstance(total, float):
            total = torch.zeros((), device=ctx.pred_ab.device)
        return total, log


__all__ = ["LossSet", "LossContext", "build_loss"]

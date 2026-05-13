"""Loss aggregation. Imports every loss module so registry is populated."""
from __future__ import annotations

import torch

from refine.config import LossConfig

from . import colorfulness as _colorfulness  # noqa: F401
from . import freq as _freq  # noqa: F401
from . import gan as _gan  # noqa: F401
from . import perceptual as _perceptual  # noqa: F401
from . import pixel as _pixel  # noqa: F401
from .gan import GeneratorGANLoss
from .registry import LossContext, build_loss


class LossSet:
    """Composes weighted losses with optional per-task masks."""

    def __init__(self, configs: list[LossConfig]):
        self.entries: list[tuple[float, object, list[str] | None]] = []
        self.has_gan = False
        self.discriminator_cfg: dict | None = None
        for c in configs:
            loss = build_loss(c.name, c.config)
            self.entries.append((float(c.weight), loss, c.apply_to_tasks))
            if isinstance(loss, GeneratorGANLoss):
                self.has_gan = True
                self.discriminator_cfg = loss.disc_config

    def parameters(self):
        for _, loss, _ in self.entries:
            for p in loss.parameters():
                yield p

    def to(self, device, dtype=None):
        for _, loss, _ in self.entries:
            loss.to(device, dtype) if dtype is not None else loss.to(device)
        return self

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        total: torch.Tensor | float = 0.0
        log: dict[str, float] = {}
        for weight, loss, mask in self.entries:
            if mask is None:
                val = loss(ctx)
            else:
                idxs = [i for i, n in enumerate(ctx.task_names) if n in mask]
                if len(idxs) == 0:
                    log[loss.name] = 0.0
                    continue
                idx_t = torch.tensor(idxs, device=ctx.pred_rgb.device)
                sub_ctx = LossContext(
                    pred_rgb=ctx.pred_rgb.index_select(0, idx_t),
                    clean_rgb=ctx.clean_rgb.index_select(0, idx_t),
                    degraded_rgb=ctx.degraded_rgb.index_select(0, idx_t),
                    task_ids=ctx.task_ids.index_select(0, idx_t),
                    task_names=[ctx.task_names[i] for i in idxs],
                    discriminator=ctx.discriminator,
                )
                val = loss(sub_ctx)
            total = total + weight * val
            log[loss.name] = float(val.detach())
        if isinstance(total, float):
            total = torch.zeros((), device=ctx.pred_rgb.device)
        return total, log


__all__ = ["LossSet", "LossContext", "build_loss"]

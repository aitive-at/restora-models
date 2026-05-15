"""Loss aggregation. Imports every loss module so registry is populated."""
from __future__ import annotations

import torch

from restora_models.config import LossConfig

from . import chroma as _chroma  # noqa: F401
from . import colorfulness as _colorfulness  # noqa: F401
from . import freq as _freq  # noqa: F401
from . import gan as _gan  # noqa: F401
from . import perceptual as _perceptual  # noqa: F401
from . import pixel as _pixel  # noqa: F401
from . import diffusion as _diffusion  # noqa: F401
from . import temporal as _temporal  # noqa: F401
from .gan import GeneratorGANLoss
from .registry import LossContext, build_loss


class LossSet:
    """Composes weighted losses with optional per-axis masks."""

    def __init__(self, configs: list[LossConfig]):
        self.entries: list[tuple[float, object, list[str] | None]] = []
        self.has_gan = False
        self.discriminator_cfg: dict | None = None
        for c in configs:
            loss = build_loss(c.name, c.config)
            self.entries.append((float(c.weight), loss, c.apply_to_axes))
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

    def __call__(self, ctx: LossContext,
                 weight_overrides: dict[str, float] | None = None
                 ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total weighted loss.

        `weight_overrides` lets the trainer dynamically scale specific
        loss weights — used for GAN warmup (ramping the 'gan' loss weight
        from 0 → configured value over N steps).
        """
        from restora_models.data.compound import AXES
        axis_to_idx = {a: i for i, a in enumerate(AXES)}

        total: torch.Tensor | float = 0.0
        log: dict[str, float] = {}
        for weight, loss, mask in self.entries:
            if weight_overrides is not None and loss.name in weight_overrides:
                weight = float(weight) * float(weight_overrides[loss.name])
            if mask is None:
                val = loss(ctx)
            else:
                # ANY of the listed axes is active for this sample
                idxs_in_mask = [axis_to_idx[a] for a in mask if a in axis_to_idx]
                if not idxs_in_mask:
                    log[loss.name] = 0.0
                    continue
                # config is float 0.0/1.0; use >= 0.5 threshold to handle both
                mask_t = (ctx.config[:, idxs_in_mask] >= 0.5).any(dim=1)
                idxs = torch.nonzero(mask_t, as_tuple=False).flatten().tolist()
                if len(idxs) == 0:
                    log[loss.name] = 0.0
                    continue
                idx_t = torch.tensor(idxs, device=ctx.pred_rgb.device)
                sub_ctx = LossContext(
                    pred_rgb=ctx.pred_rgb.index_select(0, idx_t),
                    clean_rgb=ctx.clean_rgb.index_select(0, idx_t),
                    degraded_rgb=ctx.degraded_rgb.index_select(0, idx_t),
                    config=ctx.config.index_select(0, idx_t),
                    axes_active=[ctx.axes_active[i] for i in idxs],
                    discriminator=ctx.discriminator,
                    secondary_pred_rgb=(ctx.secondary_pred_rgb.index_select(0, idx_t)
                                        if ctx.secondary_pred_rgb is not None else None),
                    flow_t_to_secondary=(ctx.flow_t_to_secondary.index_select(0, idx_t)
                                         if ctx.flow_t_to_secondary is not None else None),
                )
                val = loss(sub_ctx)
            total = total + weight * val
            log[loss.name] = float(val.detach())
        if isinstance(total, float):
            total = torch.zeros((), device=ctx.pred_rgb.device)
        return total, log


__all__ = ["LossSet", "LossContext", "build_loss"]

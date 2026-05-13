"""Fp32 ModelEMA shadow."""
from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).float()
        self.module.train(False)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        esd = self.module.state_dict()
        for k, ev in esd.items():
            mv = msd[k].detach()
            if ev.dtype.is_floating_point:
                ev.mul_(self.decay).add_(mv.to(ev.dtype), alpha=1.0 - self.decay)
            else:
                ev.copy_(mv)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)

"""VGG16-BN perceptual loss with lazy weight load and fp32 dispatch."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from .registry import LossContext, RestorationLoss, register_loss

_LAYER_INDICES = {
    "conv1_1": 0, "conv1_2": 3,
    "conv2_1": 7, "conv2_2": 10,
    "conv3_1": 14, "conv3_2": 17, "conv3_3": 20,
    "conv4_1": 24, "conv4_2": 27, "conv4_3": 30,
    "conv5_1": 34, "conv5_2": 37, "conv5_3": 40,
}


@register_loss("perceptual_vgg16bn")
class VGG16BNPerceptualLoss(RestorationLoss):
    def __init__(self, layer_weights: Mapping[str, float] | None = None,
                 criterion: str = "l1", use_input_norm: bool = True) -> None:
        super().__init__()
        from torchvision.models import VGG16_BN_Weights, vgg16_bn

        if layer_weights is None:
            layer_weights = {"conv1_1": 0.0625, "conv2_1": 0.125, "conv3_1": 0.25,
                             "conv4_1": 0.5, "conv5_1": 1.0}
        self._weights = dict(layer_weights)
        self._input_norm = bool(use_input_norm)

        feats = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        feats.train(False)
        for p in feats.parameters():
            p.requires_grad_(False)

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        last = 0
        for name in sorted(self._weights, key=lambda k: _LAYER_INDICES[k]):
            idx = _LAYER_INDICES[name]
            stages[name] = nn.Sequential(*list(feats[last : idx + 1]))
            last = idx + 1
        self._stages = nn.ModuleDict(stages)
        self._criterion = nn.L1Loss() if criterion == "l1" else nn.MSELoss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._input_norm:
            x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        out: dict[str, torch.Tensor] = {}
        for name, stage in self._stages.items():
            x = stage(x)
            out[name] = x
        return out

    def forward(self, ctx: LossContext) -> torch.Tensor:
        device_type = ctx.pred_rgb.device.type
        with torch.amp.autocast(device_type, enabled=False):
            pred_f = self._features(ctx.pred_rgb.float())
            with torch.no_grad():
                gt_f = self._features(ctx.clean_rgb.float())
            perc: torch.Tensor | float = 0.0
            for name, w in self._weights.items():
                perc = perc + w * self._criterion(pred_f[name], gt_f[name].detach())
        return perc

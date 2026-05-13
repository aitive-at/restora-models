"""VGG16-BN perceptual loss with lazy weight load."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from .registry import ColorizationLoss, LossContext, register_loss

_LAYER_INDICES = {
    "conv1_1": 0, "conv1_2": 3,
    "conv2_1": 7, "conv2_2": 10,
    "conv3_1": 14, "conv3_2": 17, "conv3_3": 20,
    "conv4_1": 24, "conv4_2": 27, "conv4_3": 30,
    "conv5_1": 34, "conv5_2": 37, "conv5_3": 40,
}


def _gram(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    f = x.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


@register_loss("perceptual_vgg16bn")
class VGG16BNPerceptualLoss(ColorizationLoss):
    def __init__(
        self,
        layer_weights: Mapping[str, float] | None = None,
        criterion: str = "l1",
        style_weight: float = 0.0,
        use_input_norm: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import VGG16_BN_Weights, vgg16_bn

        if layer_weights is None:
            layer_weights = {
                "conv1_1": 0.0625,
                "conv2_1": 0.125,
                "conv3_1": 0.25,
                "conv4_1": 0.5,
                "conv5_1": 1.0,
            }
        self._weights = dict(layer_weights)
        self.style_weight = float(style_weight)
        self._input_norm = bool(use_input_norm)

        vgg_features = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        vgg_features.train(False)
        for p in vgg_features.parameters():
            p.requires_grad_(False)

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        last = 0
        for name in sorted(self._weights, key=lambda k: _LAYER_INDICES[k]):
            idx = _LAYER_INDICES[name]
            stages[name] = nn.Sequential(*list(vgg_features[last : idx + 1]))
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
        # VGG cascades overflow bf16/fp16 dynamic range; always run perceptual in fp32.
        device_type = ctx.pred_rgb.device.type
        with torch.amp.autocast(device_type, enabled=False):
            pred_rgb = ctx.pred_rgb.float()
            gt_rgb = ctx.gt_rgb.float()
            pred_f = self._features(pred_rgb)
            with torch.no_grad():
                gt_f = self._features(gt_rgb)
            perc: torch.Tensor | float = 0.0
            for name, w in self._weights.items():
                perc = perc + w * self._criterion(pred_f[name], gt_f[name].detach())
            if self.style_weight > 0:
                sty: torch.Tensor | float = 0.0
                for name, w in self._weights.items():
                    sty = sty + w * self._criterion(
                        _gram(pred_f[name]), _gram(gt_f[name].detach())
                    )
                perc = perc + self.style_weight * sty
        return perc

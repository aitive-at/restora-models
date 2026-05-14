import torch

from restora_models.losses.freq import FreqL1Loss
from restora_models.losses.registry import LossContext


def test_freq_zero_when_equal():
    rgb = torch.rand(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=rgb, clean_rgb=rgb.clone(), degraded_rgb=z,
                      config=torch.zeros(1, 5),
                      axes_active=["sharpen"])
    assert FreqL1Loss()(ctx).item() < 1e-5


def test_freq_grad():
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    clean = torch.rand(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=pred, clean_rgb=clean, degraded_rgb=z,
                      config=torch.zeros(1, 5),
                      axes_active=["sharpen"])
    FreqL1Loss()(ctx).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0

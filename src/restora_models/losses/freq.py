"""L1 between FFT log-magnitude spectra of pred and clean."""
from __future__ import annotations

import torch

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("freq_l1")
class FreqL1Loss(RestorationLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        with torch.amp.autocast(ctx.pred_rgb.device.type, enabled=False):
            pred = torch.fft.rfft2(ctx.pred_rgb.float(), norm="ortho")
            clean = torch.fft.rfft2(ctx.clean_rgb.float(), norm="ortho")
            # Compute magnitude via real-valued ops instead of `pred.abs()`.
            # `.abs()` on a complex tensor triggers PyTorch's nvrtc kernel
            # codegen for `abs_kernel<std::complex<float>>`, which fails on
            # GPUs whose compute capability isn't in the installed PyTorch's
            # dispatch table (B300 sm_103, etc.). Doing it as
            # sqrt(re^2 + im^2) only ever generates real-valued kernels.
            # The +1e-12 inside the sqrt keeps the gradient finite at |z|=0
            # (the manual sqrt has an infinite-grad singularity there; the
            # complex .abs() has a backward special-case for the same).
            pred_mag = torch.log1p(
                torch.sqrt(pred.real.square() + pred.imag.square() + 1e-12)
                + self.eps)
            clean_mag = torch.log1p(
                torch.sqrt(clean.real.square() + clean.imag.square() + 1e-12)
                + self.eps)
            return (pred_mag - clean_mag).abs().mean()

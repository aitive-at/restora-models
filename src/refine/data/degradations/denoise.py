"""Denoising: add Gaussian noise (optionally + Poisson) with random sigma."""
from __future__ import annotations

import random

import numpy as np

from .registry import Degradation, register_degradation


@register_degradation("denoise")
class DenoiseDegradation(Degradation):
    def __init__(
        self,
        sigma_range: tuple[float, float] = (0.005, 0.05),
        poisson_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.sigma_min, self.sigma_max = float(sigma_range[0]), float(sigma_range[1])
        self.poisson_prob = float(poisson_prob)

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        sigma = rng.uniform(self.sigma_min, self.sigma_max)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        noise = np_rng.normal(0.0, sigma, rgb.shape).astype(np.float32)
        out = rgb + noise
        if rng.random() < self.poisson_prob:
            poisson_noise = np_rng.normal(0.0, 0.01 * np.sqrt(np.clip(rgb, 1e-3, 1.0)),
                                          rgb.shape).astype(np.float32)
            out = out + poisson_noise
        return np.clip(out, 0.0, 1.0).astype(np.float32)

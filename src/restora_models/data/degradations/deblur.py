"""Deblur degradation: Gaussian blur (optional motion blur)."""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    k /= k.sum() + 1e-8
    return k


@register_degradation("deblur")
class DeblurDegradation(Degradation):
    def __init__(
        self,
        sigma_range: tuple[float, float] = (1.0, 3.0),
        motion_prob: float = 0.2,
        motion_size_range: tuple[int, int] = (7, 21),
    ) -> None:
        super().__init__()
        self.sigma_min, self.sigma_max = float(sigma_range[0]), float(sigma_range[1])
        self.motion_prob = float(motion_prob)
        self.motion_min, self.motion_max = int(motion_size_range[0]), int(motion_size_range[1])

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        if rng.random() < self.motion_prob:
            size = rng.randint(self.motion_min, self.motion_max)
            if size % 2 == 0:
                size += 1
            angle = rng.uniform(0.0, 180.0)
            kernel = _motion_kernel(size, angle)
            out = cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REFLECT)
        else:
            sigma = rng.uniform(self.sigma_min, self.sigma_max)
            ksize = max(3, int(2 * math.ceil(2 * sigma) + 1))
            out = cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                   borderType=cv2.BORDER_REFLECT)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

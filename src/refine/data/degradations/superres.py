"""Super-resolution degradation: bicubic down, then bicubic up. Same-resolution output."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


class _SRBase(Degradation):
    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, w // self.factor), max(1, h // self.factor)),
                           interpolation=cv2.INTER_CUBIC)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(up, 0.0, 1.0).astype(np.float32)


@register_degradation("sr_x2")
class SRx2(_SRBase):
    def __init__(self, factor: int = 2):
        super().__init__(factor=factor)


@register_degradation("sr_x4")
class SRx4(_SRBase):
    def __init__(self, factor: int = 4):
        super().__init__(factor=factor)

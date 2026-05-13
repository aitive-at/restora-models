"""Sharpen (SR refinement) degradation: bicubic down by random factor, then bicubic up."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


@register_degradation("sharpen")
class SharpenDegradation(Degradation):
    def __init__(self, factor_choices: list[int] | tuple[int, ...] = (2, 4, 8)) -> None:
        super().__init__()
        self.factor_choices = list(factor_choices)
        if not self.factor_choices:
            raise ValueError("factor_choices must be non-empty")

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        factor = rng.choice(self.factor_choices)
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, w // factor), max(1, h // factor)),
                           interpolation=cv2.INTER_CUBIC)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(up, 0.0, 1.0).astype(np.float32)

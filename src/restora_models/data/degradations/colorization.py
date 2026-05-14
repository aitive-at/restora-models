"""Colorization: RGB → gray-as-RGB via LAB-L (a=b=0)."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


@register_degradation("colorize")
class ColorizationDegradation(Degradation):
    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L = lab[:, :, :1]
        gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
        gray_rgb = cv2.cvtColor(cv2.cvtColor(gray_lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        return gray_rgb.astype(np.float32)

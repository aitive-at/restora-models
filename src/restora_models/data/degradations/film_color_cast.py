"""Film color-cast degradation: per-channel gamma + tint matrices.

Models sepia, cyan-fade, red-shift, bleach-bypass via random LUTs.
"""
from __future__ import annotations

import random

import numpy as np

from restora_models.data.degradations.registry import Degradation, register_degradation


_PRESETS = [
    # Sepia
    (np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)),
    # Cyan fade
    (np.array([
        [0.7, 0.0, 0.0],
        [0.1, 0.9, 0.1],
        [0.1, 0.2, 1.0],
    ], dtype=np.float32), np.array([1.2, 0.9, 0.8], dtype=np.float32)),
    # Eastman red-shift
    (np.array([
        [1.1, 0.0, 0.0],
        [0.0, 0.85, 0.0],
        [0.0, 0.0, 0.75],
    ], dtype=np.float32), np.array([0.9, 1.1, 1.2], dtype=np.float32)),
    # Bleach bypass
    (np.array([
        [0.85, 0.1, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.1, 0.85],
    ], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)),
]


@register_degradation("film_color_cast")
class FilmColorCastDegradation(Degradation):
    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        idx = rng.randrange(len(_PRESETS))
        tint, gamma = _PRESETS[idx]
        # rgb shape (H, W, 3); apply tint per pixel
        flat = rgb.reshape(-1, 3)
        out = flat @ tint.T   # (N, 3)
        out = np.clip(out, 1e-6, 1.0).reshape(rgb.shape)
        # Per-channel gamma
        out = np.power(out, gamma.reshape(1, 1, 3))
        return np.clip(out, 0.0, 1.0).astype(rgb.dtype)

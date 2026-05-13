"""Tiny transforms for the training pipeline (numpy-side, before tensorization)."""
from __future__ import annotations

import random

import numpy as np


def random_crop(rgb: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {(h, w)} smaller than crop {size}")
    top = rng.randint(0, h - size)
    left = rng.randint(0, w - size)
    return rgb[top : top + size, left : left + size]


def center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return rgb[top : top + size, left : left + size]


def hflip(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, ::-1])

"""Tests for film color cast."""
import random

import numpy as np

from restora_models.data.degradations.film_color_cast import FilmColorCastDegradation


def test_color_cast_shape_preserved():
    deg = FilmColorCastDegradation()
    img = np.random.rand(64, 64, 3).astype(np.float32)
    out = deg.degrade(img, random.Random(0))
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_color_cast_changes_image():
    deg = FilmColorCastDegradation()
    img = np.random.rand(64, 64, 3).astype(np.float32)
    out = deg.degrade(img, random.Random(0))
    diff = float(np.mean(np.abs(out - img)))
    assert diff > 0.005, f"too little change: {diff}"

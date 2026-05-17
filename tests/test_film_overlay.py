"""Tests for film_overlay degradation."""
import random

import numpy as np

from restora_models.data.degradations.film_overlay import FilmOverlayDegradation


def test_film_overlay_shape_preserved():
    deg = FilmOverlayDegradation(textures=None, alpha_range=(0.1, 0.3))
    img = np.random.rand(64, 64, 3).astype(np.float32)
    out = deg.degrade(img, random.Random(0))
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_film_overlay_no_textures_returns_input():
    deg = FilmOverlayDegradation(textures=None, alpha_range=(0.1, 0.3))
    img = np.random.rand(64, 64, 3).astype(np.float32)
    out = deg.degrade(img, random.Random(0))
    np.testing.assert_array_equal(out, img)


def test_film_overlay_with_synthetic_texture():
    """A constant-0.5 overlay at alpha=0.5 should lift a zero image off zero."""
    deg = FilmOverlayDegradation(
        textures=[np.full((64, 64), 0.5, dtype=np.float32)],
        alpha_range=(0.5, 0.5),
    )
    img = np.zeros((64, 64, 3), dtype=np.float32)
    out = deg.degrade(img, random.Random(0))
    assert out.mean() > 0.01

import random

import numpy as np
import pytest

from restora_models.data.degradations import colorization, denoise, deblur, jpeg, superres  # noqa: F401
from restora_models.data.degradations.registry import DEGRADATION_REGISTRY


@pytest.fixture
def rgb_in():
    rng = np.random.default_rng(0)
    return rng.random((48, 64, 3)).astype(np.float32)


@pytest.mark.parametrize("name,cfg", [
    ("colorize", {}),
    ("denoise",  {"sigma_range": [0.01, 0.05]}),
    ("sharpen",  {"factor_choices": [2, 4]}),
    ("deblur",   {"sigma_range": [1.0, 2.0], "motion_prob": 0.0}),
    ("jpeg",     {"quality_range": [40, 60]}),
])
def test_degradation_preserves_shape_and_dtype(name, cfg, rgb_in):
    d_cls = DEGRADATION_REGISTRY[name]
    d = d_cls(**cfg)
    rng = random.Random(0)
    out = d.degrade(rgb_in.copy(), rng)
    assert out.shape == rgb_in.shape
    assert out.dtype == rgb_in.dtype
    assert out.min() >= 0.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5


def test_colorization_zeros_chroma(rgb_in):
    import cv2

    out = DEGRADATION_REGISTRY["colorize"]().degrade(rgb_in.copy(), random.Random(0))
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    assert abs(lab[:, :, 1]).mean() < 1.0
    assert abs(lab[:, :, 2]).mean() < 1.0


def test_denoise_adds_noise(rgb_in):
    d = DEGRADATION_REGISTRY["denoise"](sigma_range=[0.03, 0.03])
    out = d.degrade(rgb_in.copy(), random.Random(0))
    assert (out - rgb_in).std() > 0.01


def test_sharpen_loses_detail(rgb_in):
    d = DEGRADATION_REGISTRY["sharpen"](factor_choices=[4])
    out = d.degrade(rgb_in.copy(), random.Random(0))
    grad_in = np.abs(np.diff(rgb_in, axis=0)).mean()
    grad_out = np.abs(np.diff(out, axis=0)).mean()
    assert grad_out < grad_in

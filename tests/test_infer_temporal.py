"""Tests for the temporal VideoPipeline."""
from pathlib import Path

import cv2
import numpy as np
import torch

from restora_models.config import ModelConfig
from restora_models.infer.pipeline import VideoPipeline
from restora_models.models.registry import build_model


def _tiny_model():
    m = build_model(ModelConfig(type="temporal_restora_nano"), num_axes=5)
    m.train(False)
    return m


def test_pipeline_single_image(tmp_path):
    pipe = VideoPipeline(_tiny_model(), device="cpu")
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = pipe.process_image(img, config={"colorize": True})
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_pipeline_directory(tmp_path):
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"f{i:03d}.png"),
                    (np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    pipe = VideoPipeline(_tiny_model(), device="cpu")
    out_dir = tmp_path / "out"
    pipe.process_directory(tmp_path, out_dir, config={"colorize": True})
    outs = sorted(out_dir.glob("*.png"))
    # 10 input frames in -> 10 output files; tmp_path has subdirs filtered
    assert len(outs) == 10


def test_pipeline_handles_non_multiple_of_16(tmp_path):
    """Image with H=W=63 (not divisible by 16) should still work via padding."""
    pipe = VideoPipeline(_tiny_model(), device="cpu")
    img = (np.random.rand(63, 63, 3) * 255).astype(np.uint8)
    out = pipe.process_image(img, config={"denoise": True})
    assert out.shape == img.shape

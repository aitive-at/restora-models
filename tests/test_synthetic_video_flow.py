"""Tests for the synthetic video generator's flow direction.

Generated videos are accompanied by analytical backward flow (tk -> t).
flow_warp(frame_t, flow) should yield approximately frame_tk.
"""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from restora_models.losses.temporal import flow_warp


def _load_synth_module():
    """Load scripts/make_synthetic_videos.py as an importable module."""
    repo_root = Path(__file__).parent.parent
    script = repo_root / "scripts" / "make_synthetic_videos.py"
    spec = importlib.util.spec_from_file_location("make_synthetic_videos",
                                                    str(script))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_synthetic_videos"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def synth_mod():
    return _load_synth_module()


def test_flow_between_pure_translation(synth_mod):
    """For a pure translation by (tx, ty), the backward flow at any
    pixel p should be exactly (-tx, -ty)."""
    H = W = 32
    tx, ty = 3.0, 0.0
    cx, cy = W / 2.0, H / 2.0
    M_t = synth_mod._affine_matrix(0.0, 0.0, 0.0, (cx, cy))
    M_tk = synth_mod._affine_matrix(tx, ty, 0.0, (cx, cy))
    flow = synth_mod._flow_between(M_t, M_tk, H, W)
    assert flow.shape == (2, H, W)
    # Mean flow x ≈ -tx; mean flow y ≈ -ty (because backward flow says
    # "where did the content at p in tk come from in t" = p - translation).
    assert np.allclose(flow[0].mean(), -tx, atol=0.01)
    assert np.allclose(flow[1].mean(), -ty, atol=0.01)


def test_flow_warp_recovers_translated_frame(synth_mod):
    """Generate a (frame_t, frame_tk) pair via pure translation, then
    flow_warp(frame_t, flow) should match frame_tk within bilinear error."""
    H = W = 64
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8).astype(np.float32) / 255.0

    tx, ty = 2.0, 0.0
    cx, cy = W / 2.0, H / 2.0
    M_t = synth_mod._affine_matrix(0.0, 0.0, 0.0, (cx, cy))
    M_tk = synth_mod._affine_matrix(tx, ty, 0.0, (cx, cy))

    frame_t = synth_mod._warp_with_affine(img, M_t)
    frame_tk = synth_mod._warp_with_affine(img, M_tk)
    flow = synth_mod._flow_between(M_t, M_tk, H, W)

    # Run through flow_warp
    t = torch.from_numpy(frame_t.transpose(2, 0, 1)).unsqueeze(0)
    tk = torch.from_numpy(frame_tk.transpose(2, 0, 1)).unsqueeze(0)
    f = torch.from_numpy(flow).unsqueeze(0)

    warped = flow_warp(t, f)
    err = F.l1_loss(warped, tk).item()
    # ~0.05 is acceptable given bilinear interpolation + border handling.
    assert err < 0.08, f"L1 error {err:.4f} — flow direction may be wrong"

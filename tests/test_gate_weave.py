"""Tests for gate-weave (per-frame sub-pixel jitter)."""
import torch

from restora_models.data.degradations.gate_weave import GateWeaveDegradation


def test_gate_weave_per_frame_shape():
    deg = GateWeaveDegradation(max_shift_px=2.0)
    clip = torch.rand(7, 3, 64, 64)
    out = deg.apply_clip(clip)
    assert out.shape == clip.shape


def test_gate_weave_zero_shift_returns_input():
    deg = GateWeaveDegradation(max_shift_px=0.0)
    clip = torch.rand(7, 3, 32, 32)
    out = deg.apply_clip(clip)
    assert torch.allclose(out, clip, atol=1e-4)

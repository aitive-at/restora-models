"""Tests for flow_warp + visibility_mask in models/warp.py."""
import torch

from restora_models.models.warp import flow_warp, visibility_mask


def test_flow_warp_identity_flow_returns_input():
    rgb = torch.rand(2, 3, 32, 32)
    zero_flow = torch.zeros(2, 2, 32, 32)
    out = flow_warp(rgb, zero_flow)
    assert torch.allclose(out, rgb, atol=1e-5)


def test_flow_warp_pixel_shift():
    rgb = torch.zeros(1, 3, 8, 8)
    rgb[:, :, 4, 4] = 1.0
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0, :, :] = 1.0
    out = flow_warp(rgb, flow)
    assert out[0, 0, 4, 3].item() > 0.5


def test_visibility_mask_zero_flow_all_visible():
    zero = torch.zeros(2, 2, 16, 16)
    mask = visibility_mask(zero, zero, threshold=0.5)
    assert torch.all(mask >= 0.99)


def test_visibility_mask_inconsistent_flows_low():
    fwd = torch.zeros(1, 2, 16, 16)
    bwd = torch.zeros(1, 2, 16, 16)
    fwd[0, 0, 8, 8] = 5.0
    mask = visibility_mask(fwd, bwd, threshold=0.5)
    assert mask[0, 0, 8, 8].item() < 0.5

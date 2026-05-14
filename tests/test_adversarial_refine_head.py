"""Tests for AdversarialRefineHead."""
import torch

from restora_models.models.heads import AdversarialRefineHead


def _head(feat_dim: int = 16, hidden_dim: int = 32, n_blocks: int = 2):
    return AdversarialRefineHead(feat_dim=feat_dim, hidden_dim=hidden_dim,
                                 n_blocks=n_blocks)


def test_forward_shape():
    h = _head()
    features = torch.randn(2, 16, 32, 32)
    coarse_rgb = torch.rand(2, 3, 32, 32)
    config = torch.zeros(2, 5)
    out = h(features, coarse_rgb, config)
    assert out.shape == coarse_rgb.shape
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_initial_output_close_to_coarse():
    """At init, refine head's delta contribution is small (~5e-3 magnitude)
    so refined output ≈ coarse. Not identical because proj_out has small-
    normal init to keep gradient flowing backward."""
    h = _head()
    h.train(False)
    features = torch.randn(1, 16, 16, 16)
    coarse = torch.rand(1, 3, 16, 16)
    config = torch.tensor([[1.0, 0, 0, 0, 0]])
    with torch.no_grad():
        out = h(features, coarse, config)
    # Tiny deviation from coarse (well under 5%)
    assert (out - coarse).abs().mean().item() < 0.02


def test_residual_scale_in_range():
    """Learnable residual scale must be in [0, 0.5] via the sigmoid+scale param."""
    h = _head()
    s = h.residual_scale.item()
    assert 0.0 <= s <= 0.5
    # Pushing residual_scale_raw to ±∞ should saturate at 0 and 0.5
    with torch.no_grad():
        h.residual_scale_raw.data.fill_(20.0)
    assert h.residual_scale.item() > 0.499
    with torch.no_grad():
        h.residual_scale_raw.data.fill_(-20.0)
    assert h.residual_scale.item() < 1e-6


def test_config_drives_different_outputs():
    """Different configs should produce different refinements ONCE the
    AdaLN proj layers have moved off zero. At init they are zero (DiT
    convention) so cond has no effect — but after one optimizer step
    they should differentiate."""
    h = _head()
    # Push proj_out + at least one AdaLN proj off zero so cond is wired
    with torch.no_grad():
        h.proj_out.weight.normal_(0, 0.1)
        h.residual_scale_raw.fill_(2.0)            # ≈ 0.44 scale, near max
        for blk in h.blocks:
            blk.adaln1.proj.weight.normal_(0, 0.1)
            blk.adaln1.proj.bias.normal_(0, 0.1)
    features = torch.randn(1, 16, 16, 16)
    coarse = torch.rand(1, 3, 16, 16)
    c1 = torch.tensor([[1.0, 0, 0, 0, 0]])
    c2 = torch.tensor([[0, 0, 1.0, 0, 0]])
    h.train(False)
    with torch.no_grad():
        o1 = h(features, coarse, c1)
        o2 = h(features, coarse, c2)
    assert (o1 - o2).abs().mean().item() > 1e-4


def test_backward_main_params_get_grads():
    """proj_out (small-normal) and refine block conv weights get gradient
    at step 0 — proj_out's non-zero init lets backward signal propagate
    into earlier blocks. cond_embed + AdaLN proj are still zero-init
    (DiT convention), so they only start getting gradient after AdaLN
    proj has moved off zero — not tested at init."""
    h = _head()
    features = torch.randn(1, 16, 16, 16, requires_grad=False)
    coarse = torch.rand(1, 3, 16, 16, requires_grad=False)
    config = torch.tensor([[1.0, 0, 0, 0, 0]])
    out = h(features, coarse, config)
    out.sum().backward()
    grads = {n: p.grad for n, p in h.named_parameters()}
    assert grads["proj_out.weight"].abs().sum().item() > 0
    assert grads["blocks.0.conv1.weight"].abs().sum().item() > 0
    assert grads["residual_scale_raw"].abs().sum().item() > 0
    assert grads["proj_in.weight"].abs().sum().item() > 0


def test_param_count_in_budget():
    """Default hidden_dim=128, n_blocks=8, feat_dim=64 (NAFNet-large nf).
    Tuned for ~3M params — a fast refinement head, not a heavy generator.
    Test catches drift if defaults change."""
    h = AdversarialRefineHead(feat_dim=64)   # defaults
    n = sum(p.numel() for p in h.parameters())
    assert 1_000_000 < n < 6_000_000, f"refine head param count {n}"


def test_residual_scale_starts_near_zero():
    """At init, residual_scale should be tiny so refine head adds ~0 contribution.
    Critical for unified phase training: Phase 1 (no GAN) needs refine head
    to behave as identity. Combined with proj_out zero-init this guarantees
    initial output == coarse output."""
    h = _head()
    s = h.residual_scale.item()
    assert s < 0.05, f"residual_scale at init = {s} — should be near 0"


def test_clamps_to_valid_rgb_range():
    """Even if internal residual is large, output should be clamped to [0, 1]."""
    h = _head()
    with torch.no_grad():
        h.proj_out.bias.fill_(10.0)
        h.residual_scale_raw.fill_(20.0)
    features = torch.randn(1, 16, 16, 16)
    coarse = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out = h(features, coarse, torch.zeros(1, 5))
    assert (out >= 0.0).all() and (out <= 1.0).all()

# Latent Diffusion Refine Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the single-step latent diffusion refine head described in `docs/superpowers/specs/2026-05-16-latent-diffusion-refine-head-design.md`, replacing the existing AdversarialRefineHead while keeping the public `(rgb, config) -> rgb` model contract.

**Architecture:** A new `LatentDiffusionRefineHead` is added in `src/restora_models/models/diffusion_head.py`. It wraps a frozen SD 1.5 VAE (`stabilityai/sd-vae-ft-ema`, loaded once via `diffusers`) and a small custom AdaLN-conditioned UNet. The head encodes the deterministic dual-head output to VAE latents, interpolates between the latent and Gaussian noise per a single-step linear schedule, predicts the clean latent via the UNet, then decodes back to RGB. `NAFNetMultiTask` selects between the adversarial head and the diffusion head via a new `cfg.model.refine_type` config field.

**Tech Stack:** PyTorch >=2.4, diffusers >=0.30 (new dep), torchvision, existing test harness (pytest). Trains from the production checkpoint produced by `configs/b200.yaml`. Same I/O contract for ONNX export and downstream C# consumer.

---

## File Structure

**Create:**
- `src/restora_models/models/vae.py` — Frozen SD 1.5 VAE wrapper (`FrozenSD15VAE`)
- `src/restora_models/models/diffusion_head.py` — `LatentDiffusionRefineHead` + supporting blocks (sinusoidal timestep embed, AdaLN-resblock, conditional UNet)
- `src/restora_models/losses/diffusion.py` — `l1_latent` loss
- `configs/b200-diffusion.yaml` — Stage 1 training config (head-only, frozen backbone)
- `tests/test_frozen_vae.py`
- `tests/test_diffusion_head_blocks.py`
- `tests/test_diffusion_head.py`
- `tests/test_l1_latent_loss.py`
- `tests/test_nafnet_diffusion_wiring.py`

**Modify:**
- `pyproject.toml` — add `diffusers>=0.30` to dependencies
- `src/restora_models/config.py` — add `refine_type: Literal["none", "adversarial", "diffusion"]` to `ModelConfig`; add `diffusion_t_inference: float = 0.2` field
- `src/restora_models/losses/registry.py` — extend `LossContext` with `pred_latent` and `target_latent` (both `torch.Tensor | None`, default `None`)
- `src/restora_models/losses/__init__.py` — import the new `diffusion` losses module
- `src/restora_models/models/nafnet.py` — instantiate the right refine head based on `cfg.model.refine_type`; in forward, when diffusion head is active, expose `pred_latent` and `target_latent` via a return-extras dict
- `src/restora_models/train/trainer.py` — when diffusion head is active, encode `clean_rgb` to `target_latent` once per step; populate `LossContext.pred_latent` / `target_latent` from the model's extras
- `tests/test_configs_load.py` — add a test that `b200-diffusion.yaml` parses

---

## Task 1: Add diffusers dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add diffusers dep to pyproject.toml**

Edit `pyproject.toml`. Inside the `dependencies = [...]` list, after the `"huggingface_hub>=0.24",` line, add:

```toml
  "diffusers>=0.30",
```

- [ ] **Step 2: Sync the venv**

Run: `uv sync`
Expected: no errors; new packages installed (diffusers + transformers + others).

- [ ] **Step 3: Verify diffusers can load the SD VAE (one-time download)**

Run:
```sh
uv run python -c "from diffusers import AutoencoderKL; v = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema'); print('vae loaded, params:', sum(p.numel() for p in v.parameters()))"
```
Expected: prints `vae loaded, params: 83653863` (or similar ~80M). Caches to `~/.cache/huggingface/hub/`.

- [ ] **Step 4: Verify nothing else broke**

Run: `uv run pytest -q`
Expected: 151 passed / 6 skipped (same as before).

- [ ] **Step 5: Commit**

```sh
git add pyproject.toml uv.lock
git commit -m "deps: add diffusers>=0.30 for latent diffusion refine head"
```

---

## Task 2: FrozenSD15VAE wrapper

**Files:**
- Create: `src/restora_models/models/vae.py`
- Create: `tests/test_frozen_vae.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_frozen_vae.py`:

```python
"""Tests for the FrozenSD15VAE wrapper."""
import pytest
import torch


def test_vae_encode_returns_4ch_latent_at_8x_downsample():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    rgb = torch.rand(2, 3, 256, 256)
    z = vae.encode(rgb)
    assert z.shape == (2, 4, 32, 32), f"got {tuple(z.shape)}"
    assert -3 < z.mean().item() < 3
    assert 0.1 < z.std().item() < 3


def test_vae_decode_returns_3ch_rgb_at_8x_upsample():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    z = torch.randn(2, 4, 32, 32) * 0.5
    rgb = vae.decode(z)
    assert rgb.shape == (2, 3, 256, 256)
    assert 0.0 <= rgb.min().item() and rgb.max().item() <= 1.001


def test_vae_encode_decode_roundtrip_psnr():
    """A clean roundtrip on natural-looking content should have PSNR > 25 dB."""
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    h = w = 256
    xs = torch.linspace(0, 1, w).unsqueeze(0).unsqueeze(0).expand(3, h, w)
    ys = torch.linspace(0, 1, h).unsqueeze(0).unsqueeze(-1).expand(3, h, w)
    rgb_in = ((xs + ys) / 2.0).unsqueeze(0).clamp(0, 1)
    rgb_out = vae.decode(vae.encode(rgb_in)).clamp(0, 1)
    mse = ((rgb_in - rgb_out) ** 2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-9))).item()
    assert psnr > 25.0, f"VAE roundtrip too lossy: {psnr:.2f} dB"


def test_vae_params_are_frozen():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    assert all(not p.requires_grad for p in vae.parameters())


def test_vae_encode_mode_is_deterministic():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    rgb = torch.rand(1, 3, 128, 128)
    z1 = vae.encode_mode(rgb)
    z2 = vae.encode_mode(rgb)
    assert torch.allclose(z1, z2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_frozen_vae.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement FrozenSD15VAE**

Create `src/restora_models/models/vae.py`:

```python
"""Frozen wrapper around Stability AI's SD 1.5 VAE (sd-vae-ft-ema).

Used by the latent diffusion refine head — encode the deterministic
coarse output to a 4-channel latent, run the diffusion UNet there, then
decode back. The VAE weights are frozen; never train them.
"""
from __future__ import annotations

import torch
from torch import nn

# Canonical SD 1.5 latent scale.
_SCALE = 0.18215


class FrozenSD15VAE(nn.Module):
    """Wraps `diffusers.AutoencoderKL` from `stabilityai/sd-vae-ft-ema`,
    freezes all weights, exposes encode()/decode() in the RGB [0,1] domain.

    Inputs/outputs:
      encode(rgb_01: (B,3,H,W) in [0,1]) -> z: (B,4,H/8,W/8)
      decode(z: (B,4,H/8,W/8)) -> rgb_01: (B,3,H,W) in [0,1]

    H and W must be multiples of 8 (VAE downsample factor).
    """

    def __init__(self) -> None:
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.train(False)

    @torch.no_grad()
    def encode(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Stochastic encode (samples from the latent distribution).
        Use for training where noise in the target is standard SD practice."""
        x = rgb_01 * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.sample() * _SCALE
        return z

    @torch.no_grad()
    def encode_mode(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Deterministic encode (returns the latent distribution's mean).
        Use for inference where we want reproducible outputs."""
        x = rgb_01 * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.mode() * _SCALE
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        rgb_m11 = self.vae.decode(z / _SCALE).sample
        return (rgb_m11 + 1.0) / 2.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_frozen_vae.py -v`
Expected: 5 passed.

(Tests download VAE weights on first run; allow ~10-30 seconds.)

- [ ] **Step 5: Commit**

```sh
git add src/restora_models/models/vae.py tests/test_frozen_vae.py
git commit -m "models: add FrozenSD15VAE wrapper"
```

---

## Task 3: Diffusion-head building blocks

**Files:**
- Create: `src/restora_models/models/diffusion_head.py`
- Create: `tests/test_diffusion_head_blocks.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diffusion_head_blocks.py`:

```python
"""Unit tests for the LatentDiffusionRefineHead building blocks."""
import torch


def test_sinusoidal_timestep_embedding_shape_and_range():
    from restora_models.models.diffusion_head import sinusoidal_timestep_embedding
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = sinusoidal_timestep_embedding(t, dim=64)
    assert emb.shape == (3, 64)
    assert (-1.001 <= emb).all() and (emb <= 1.001).all()


def test_sinusoidal_timestep_embedding_distinct_per_t():
    from restora_models.models.diffusion_head import sinusoidal_timestep_embedding
    t = torch.tensor([0.1, 0.2, 0.3])
    emb = sinusoidal_timestep_embedding(t, dim=128)
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[1], emb[2])


def test_adaln_resblock_preserves_shape():
    from restora_models.models.diffusion_head import AdaLNResBlock
    block = AdaLNResBlock(c=96, cond_dim=384)
    x = torch.randn(2, 96, 32, 32)
    cond = torch.randn(2, 384)
    out = block(x, cond)
    assert out.shape == x.shape


def test_adaln_resblock_has_residual_path():
    """At zero-init the residual path should be ~identity."""
    from restora_models.models.diffusion_head import AdaLNResBlock
    block = AdaLNResBlock(c=64, cond_dim=128)
    torch.nn.init.zeros_(block.conv2.weight)
    torch.nn.init.zeros_(block.conv2.bias)
    x = torch.randn(1, 64, 16, 16)
    cond = torch.randn(1, 128)
    out = block(x, cond)
    assert torch.allclose(out, x)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diffusion_head_blocks.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the building blocks**

Create `src/restora_models/models/diffusion_head.py`:

```python
"""LatentDiffusionRefineHead — single-step diffusion refinement in SD 1.5
VAE latent space.

See docs/superpowers/specs/2026-05-16-latent-diffusion-refine-head-design.md
for the design rationale.

This file contains:
  - sinusoidal_timestep_embedding: positional encoding for the diffusion t
  - AdaLNResBlock:                 conditional residual block (AdaLN + conv)
  - LatentDiffusionRefineHead:     the full head (added in a later task)
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for a continuous timestep t in [0, 1].
    Maps (B,) -> (B, dim) with values in [-1, 1]."""
    if t.dim() == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 10000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class _AdaLN(nn.Module):
    """Group-norm + per-channel scale/shift conditioned on cond (B, cond_dim)."""

    def __init__(self, c: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, c), num_channels=c, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * c)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1.0 + gamma.view(-1, x.shape[1], 1, 1)) + beta.view(-1, x.shape[1], 1, 1)


class AdaLNResBlock(nn.Module):
    """Residual block with two AdaLN-conditioned conv layers.

      h = adaLN1(x, cond) -> SiLU -> conv1
      h = adaLN2(h, cond) -> SiLU -> conv2
      out = x + h
    """

    def __init__(self, c: int, cond_dim: int) -> None:
        super().__init__()
        self.adaln1 = _AdaLN(c, cond_dim)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.adaln2 = _AdaLN(c, cond_dim)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.adaln2(h, cond)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diffusion_head_blocks.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```sh
git add src/restora_models/models/diffusion_head.py tests/test_diffusion_head_blocks.py
git commit -m "models: diffusion head building blocks (timestep emb + AdaLN-resblock)"
```

---

## Task 4: LatentDiffusionRefineHead main class

**Files:**
- Modify: `src/restora_models/models/diffusion_head.py`
- Create: `tests/test_diffusion_head.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diffusion_head.py`:

```python
"""End-to-end tests for LatentDiffusionRefineHead."""
import pytest
import torch


@pytest.fixture
def head():
    from restora_models.models.diffusion_head import LatentDiffusionRefineHead
    return LatentDiffusionRefineHead(feat_dim=64, num_axes=5)


def test_forward_inference_returns_rgb_at_input_resolution(head):
    head.train(False)
    feat = torch.randn(2, 64, 256, 256)
    coarse = torch.rand(2, 3, 256, 256)
    config = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out = head(feat, coarse, config)
    assert out.shape == (2, 3, 256, 256)
    assert 0.0 <= out.min().item() and out.max().item() <= 1.001


def test_forward_training_returns_pred_and_target_latent(head):
    head.train(True)
    feat = torch.randn(2, 64, 256, 256)
    coarse = torch.rand(2, 3, 256, 256)
    clean = torch.rand(2, 3, 256, 256)
    config = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
    pred_latent, target_latent, decoded_rgb = head.forward_with_targets(
        feat, coarse, clean, config)
    assert pred_latent.shape == (2, 4, 32, 32)
    assert target_latent.shape == (2, 4, 32, 32)
    assert decoded_rgb.shape == (2, 3, 256, 256)


def test_inference_is_deterministic_with_zero_noise(head):
    head.train(False)
    head.set_inference_noise_mode("zero")
    feat = torch.randn(1, 64, 64, 64)
    coarse = torch.rand(1, 3, 64, 64)
    config = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out1 = head(feat, coarse, config)
        out2 = head(feat, coarse, config)
    assert torch.allclose(out1, out2)


def test_param_count_in_budget():
    from restora_models.models.diffusion_head import LatentDiffusionRefineHead
    head = LatentDiffusionRefineHead(feat_dim=64, num_axes=5)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert 15_000_000 < trainable < 35_000_000, \
        f"head has {trainable:,} trainable params (target ~25M)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diffusion_head.py -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement LatentDiffusionRefineHead**

Append to `src/restora_models/models/diffusion_head.py`:

```python


from .vae import FrozenSD15VAE


def _down(in_c: int, out_c: int) -> nn.Module:
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)


def _up(in_c: int, out_c: int) -> nn.Module:
    return nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)


class LatentDiffusionRefineHead(nn.Module):
    """Single-step diffusion refine head in SD 1.5 VAE latent space.

    Inference forward:
        z_coarse        = VAE.encode_mode(coarse_rgb)
        eps             = noise_for(z_coarse)
        z_t             = (1 - t_inf) * z_coarse + t_inf * eps
        feat_at_latent  = feat_proj(backbone_features)
        pred_z_clean    = unet(z_t, z_coarse, feat_at_latent, cond)
        refined_rgb     = VAE.decode(pred_z_clean)

    Training forward (forward_with_targets):
        Same as inference but:
          - t is sampled from Uniform(0, 1) per-sample
          - clean_rgb is provided so we can also compute z_clean
          - returns (pred_z_clean, z_clean, decoded_rgb)
    """

    def __init__(
        self,
        *,
        feat_dim: int = 64,
        num_axes: int = 5,
        base_c: int = 96,
        num_blocks_per_stage: int = 2,
        timestep_emb_dim: int = 128,
        t_inference: float = 0.2,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_axes = num_axes
        self.t_inference = float(t_inference)
        self._noise_mode = "random"
        self._fixed_noise: torch.Tensor | None = None

        self.vae = FrozenSD15VAE()

        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, 32, kernel_size=8, stride=8),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=1),
        )

        self._cond_dim = base_c * 4
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_axes + timestep_emb_dim, self._cond_dim),
            nn.SiLU(),
            nn.Linear(self._cond_dim, self._cond_dim),
        )
        self._timestep_emb_dim = timestep_emb_dim

        c = base_c
        self.stem = nn.Conv2d(4 + 4 + 16, c, kernel_size=1)

        self.enc1 = nn.ModuleList([AdaLNResBlock(c, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])
        self.down1 = _down(c, c * 2)

        self.enc2 = nn.ModuleList([AdaLNResBlock(c * 2, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])
        self.down2 = _down(c * 2, c * 4)

        self.bottle = nn.ModuleList([AdaLNResBlock(c * 4, self._cond_dim)
                                      for _ in range(num_blocks_per_stage * 2)])

        self.up2 = _up(c * 4, c * 2)
        self.dec2 = nn.ModuleList([AdaLNResBlock(c * 2, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])

        self.up1 = _up(c * 2, c)
        self.dec1 = nn.ModuleList([AdaLNResBlock(c, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])

        self.head_out = nn.Conv2d(c, 4, kernel_size=3, padding=1)
        nn.init.normal_(self.head_out.weight, std=0.01)
        nn.init.zeros_(self.head_out.bias)

    def set_inference_noise_mode(self, mode: str,
                                  fixed: torch.Tensor | None = None) -> None:
        """Choose how noise is sampled at inference time.

        Modes:
          'random': torch.randn_like at each call.
          'zero':   noise = 0 (degenerate; VAE-decodes z_coarse).
          'fixed':  use a pre-supplied noise tensor.
        """
        if mode not in ("random", "zero", "fixed"):
            raise ValueError(f"unknown noise mode {mode!r}")
        if mode == "fixed" and fixed is None:
            raise ValueError("noise_mode='fixed' requires a fixed= tensor")
        self._noise_mode = mode
        self._fixed_noise = fixed

    def _noise_for(self, z: torch.Tensor) -> torch.Tensor:
        if self._noise_mode == "zero":
            return torch.zeros_like(z)
        if self._noise_mode == "fixed":
            return self._fixed_noise.to(z.device, z.dtype).expand_as(z)
        return torch.randn_like(z)

    def _condition(self, config: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_timestep_embedding(t, self._timestep_emb_dim)
        return self.cond_mlp(torch.cat([config, t_emb], dim=-1))

    def _unet(self, z_t: torch.Tensor, z_coarse: torch.Tensor,
              feat_at_latent: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.stem(torch.cat([z_t, z_coarse, feat_at_latent], dim=1))
        for b in self.enc1:
            x = b(x, cond)
        skip1 = x
        x = self.down1(x)
        for b in self.enc2:
            x = b(x, cond)
        skip2 = x
        x = self.down2(x)
        for b in self.bottle:
            x = b(x, cond)
        x = self.up2(x)
        x = x + skip2
        for b in self.dec2:
            x = b(x, cond)
        x = self.up1(x)
        x = x + skip1
        for b in self.dec1:
            x = b(x, cond)
        return self.head_out(x)

    def forward(self, backbone_features: torch.Tensor,
                coarse_rgb: torch.Tensor,
                config: torch.Tensor) -> torch.Tensor:
        z_coarse = self.vae.encode_mode(coarse_rgb)
        eps = self._noise_for(z_coarse)
        t = torch.full((coarse_rgb.shape[0],),
                       self.t_inference, device=coarse_rgb.device)
        z_t = (1.0 - self.t_inference) * z_coarse + self.t_inference * eps
        feat = self.feat_proj(backbone_features)
        cond = self._condition(config, t)
        pred_z = self._unet(z_t, z_coarse, feat, cond)
        return self.vae.decode(pred_z).clamp(0.0, 1.0)

    def forward_with_targets(
        self,
        backbone_features: torch.Tensor,
        coarse_rgb: torch.Tensor,
        clean_rgb: torch.Tensor,
        config: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-time forward. Returns (pred_z_clean, target_z_clean, decoded_rgb)."""
        B = coarse_rgb.shape[0]
        z_coarse = self.vae.encode_mode(coarse_rgb)
        target_z_clean = self.vae.encode(clean_rgb)
        eps = torch.randn_like(z_coarse)
        t = torch.rand(B, device=coarse_rgb.device)
        t_b = t.view(B, 1, 1, 1)
        z_t = (1.0 - t_b) * z_coarse + t_b * eps
        feat = self.feat_proj(backbone_features)
        cond = self._condition(config, t)
        pred_z_clean = self._unet(z_t, z_coarse, feat, cond)
        decoded_rgb = self.vae.decode(pred_z_clean).clamp(0.0, 1.0)
        return pred_z_clean, target_z_clean, decoded_rgb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diffusion_head.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```sh
git add src/restora_models/models/diffusion_head.py tests/test_diffusion_head.py
git commit -m "models: LatentDiffusionRefineHead main class"
```

---

## Task 5: l1_latent loss + LossContext extension

**Files:**
- Modify: `src/restora_models/losses/registry.py`
- Create: `src/restora_models/losses/diffusion.py`
- Modify: `src/restora_models/losses/__init__.py`
- Create: `tests/test_l1_latent_loss.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_l1_latent_loss.py`:

```python
"""Tests for the l1_latent diffusion loss."""
import torch

from restora_models.losses.registry import LossContext, build_loss


def _ctx_with_latents(B=2, H=256, W=256):
    return LossContext(
        pred_rgb=torch.zeros(B, 3, H, W),
        clean_rgb=torch.zeros(B, 3, H, W),
        degraded_rgb=torch.zeros(B, 3, H, W),
        config=torch.zeros(B, 5),
        axes_active=["identity"] * B,
        pred_latent=torch.randn(B, 4, H // 8, W // 8),
        target_latent=torch.randn(B, 4, H // 8, W // 8),
    )


def test_l1_latent_returns_scalar():
    loss = build_loss("l1_latent")
    ctx = _ctx_with_latents()
    out = loss(ctx)
    assert out.dim() == 0


def test_l1_latent_zero_when_pred_equals_target():
    loss = build_loss("l1_latent")
    ctx = _ctx_with_latents()
    ctx.pred_latent = ctx.target_latent.clone()
    out = loss(ctx)
    assert out.item() == 0.0


def test_l1_latent_returns_zero_when_latents_absent():
    loss = build_loss("l1_latent")
    ctx = LossContext(
        pred_rgb=torch.zeros(1, 3, 64, 64),
        clean_rgb=torch.zeros(1, 3, 64, 64),
        degraded_rgb=torch.zeros(1, 3, 64, 64),
        config=torch.zeros(1, 5),
        axes_active=["identity"],
    )
    out = loss(ctx)
    assert out.item() == 0.0


def test_l1_latent_positive_when_latents_differ():
    loss = build_loss("l1_latent")
    ctx = _ctx_with_latents()
    out = loss(ctx)
    assert out.item() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_l1_latent_loss.py -v`
Expected: FAIL.

- [ ] **Step 3: Extend LossContext with latent fields**

Edit `src/restora_models/losses/registry.py`. Replace the existing `LossContext` dataclass body with:

```python
@dataclass
class LossContext:
    pred_rgb: torch.Tensor
    clean_rgb: torch.Tensor
    degraded_rgb: torch.Tensor
    config: torch.Tensor
    axes_active: list[str]
    discriminator: nn.Module | None = None
    secondary_pred_rgb: torch.Tensor | None = None
    flow_t_to_secondary: torch.Tensor | None = None
    # Diffusion-head intermediates. Trainer populates these when
    # cfg.model.refine_type == "diffusion"; absent otherwise.
    pred_latent: torch.Tensor | None = None
    target_latent: torch.Tensor | None = None
```

- [ ] **Step 4: Implement l1_latent loss**

Create `src/restora_models/losses/diffusion.py`:

```python
"""Diffusion training losses for the LatentDiffusionRefineHead."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("l1_latent")
class L1LatentLoss(RestorationLoss):
    """L1 between predicted and target VAE latents. Returns 0 if either
    field is missing (no-op on non-diffusion batches)."""

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.pred_latent is None or ctx.target_latent is None:
            return torch.zeros((), device=ctx.pred_rgb.device,
                                dtype=ctx.pred_rgb.dtype)
        return F.l1_loss(ctx.pred_latent, ctx.target_latent)
```

- [ ] **Step 5: Register the loss module**

Edit `src/restora_models/losses/__init__.py`. After `from . import pixel as _pixel  # noqa: F401`, add:

```python
from . import diffusion as _diffusion  # noqa: F401
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_l1_latent_loss.py -v`
Expected: 4 passed.

- [ ] **Step 7: Run full suite**

Run: `uv run pytest -q`
Expected: 159+ passed.

- [ ] **Step 8: Commit**

```sh
git add src/restora_models/losses/registry.py src/restora_models/losses/diffusion.py src/restora_models/losses/__init__.py tests/test_l1_latent_loss.py
git commit -m "losses: add l1_latent for diffusion-head supervision"
```

---

## Task 6: ModelConfig.refine_type + diffusion_t_inference

**Files:**
- Modify: `src/restora_models/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/test_config.py`. At the bottom, add:

```python
def test_model_config_refine_type_defaults():
    from restora_models.config import ModelConfig
    m = ModelConfig()
    assert m.refine_type == "none"
    assert m.diffusion_t_inference == 0.2


def test_model_config_refine_type_accepts_diffusion():
    from restora_models.config import ModelConfig
    m = ModelConfig(refine_type="diffusion", diffusion_t_inference=0.3)
    assert m.refine_type == "diffusion"
    assert m.diffusion_t_inference == 0.3


def test_model_config_legacy_adversarial_refine_coerces_to_refine_type():
    from restora_models.config import ModelConfig
    m = ModelConfig(adversarial_refine=True)
    assert m.refine_type == "adversarial"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::test_model_config_refine_type_defaults -v`
Expected: FAIL.

- [ ] **Step 3: Add the fields to ModelConfig**

Edit `src/restora_models/config.py`. Change the imports line to include `model_validator`:

```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

In `ModelConfig`, replace the `adversarial_refine` block with:

```python
    # Refine head type. Three options:
    #   "none"        - just the deterministic dual-head output
    #   "adversarial" - AdversarialRefineHead trained with GAN (current production)
    #   "diffusion"   - LatentDiffusionRefineHead in SD 1.5 VAE latent space
    refine_type: Literal["none", "adversarial", "diffusion"] = "none"
    # Legacy: implies refine_type="adversarial" when True. Coerced via
    # model_validator below for back-compat with old configs / ckpts.
    adversarial_refine: bool = False
    refine_hidden_dim: int | None = None
    refine_n_blocks: int | None = None
    # Diffusion-head-specific (ignored when refine_type != "diffusion")
    diffusion_t_inference: float = 0.2
```

After the field definitions in `ModelConfig`, add:

```python
    @model_validator(mode="after")
    def _coerce_legacy_adversarial_refine(self):
        if self.adversarial_refine and self.refine_type == "none":
            self.refine_type = "adversarial"
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: 162+ passed.

- [ ] **Step 6: Commit**

```sh
git add src/restora_models/config.py tests/test_config.py
git commit -m "config: add ModelConfig.refine_type + diffusion_t_inference"
```

---

## Task 7: Wire diffusion head into NAFNetMultiTask

**Files:**
- Modify: `src/restora_models/models/nafnet.py`
- Create: `tests/test_nafnet_diffusion_wiring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nafnet_diffusion_wiring.py`:

```python
"""Tests for NAFNetMultiTask using the diffusion refine head."""
import torch

from restora_models.config import ModelConfig
from restora_models.models import build_model


def test_nafnet_with_refine_type_none_returns_coarse_only():
    cfg = ModelConfig(type="nafnet", size="tiny", refine_type="none", input_size=64)
    model = build_model(cfg, num_axes=5)
    rgb = torch.rand(1, 3, 64, 64)
    config = torch.zeros(1, 5)
    out = model(rgb, config)
    assert out.shape == (1, 3, 64, 64)


def test_nafnet_with_diffusion_refine_returns_correct_shape():
    cfg = ModelConfig(type="nafnet", size="tiny",
                       refine_type="diffusion", input_size=64)
    model = build_model(cfg, num_axes=5)
    model.train(False)
    rgb = torch.rand(1, 3, 64, 64)
    config = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out = model(rgb, config)
    assert out.shape == (1, 3, 64, 64)


def test_nafnet_with_diffusion_exposes_latents_via_forward_with_extras():
    cfg = ModelConfig(type="nafnet", size="tiny",
                       refine_type="diffusion", input_size=64)
    model = build_model(cfg, num_axes=5)
    rgb = torch.rand(1, 3, 64, 64)
    clean = torch.rand(1, 3, 64, 64)
    config = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    pred_rgb, extras = model.forward_with_extras(rgb, clean, config)
    assert pred_rgb.shape == (1, 3, 64, 64)
    assert "pred_latent" in extras and "target_latent" in extras
    assert extras["pred_latent"].shape == (1, 4, 8, 8)
    assert extras["target_latent"].shape == (1, 4, 8, 8)


def test_nafnet_legacy_adversarial_refine_still_works():
    cfg = ModelConfig(type="nafnet", size="tiny",
                       adversarial_refine=True, input_size=64)
    model = build_model(cfg, num_axes=5)
    rgb = torch.rand(1, 3, 64, 64)
    config = torch.zeros(1, 5)
    out = model(rgb, config)
    assert out.shape == (1, 3, 64, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nafnet_diffusion_wiring.py -v`
Expected: FAIL.

- [ ] **Step 3: Modify NAFNetMultiTask to support refine_type**

Edit `src/restora_models/models/nafnet.py`. Replace the existing refine-head instantiation block at the end of `__init__` with:

```python
        if cfg.refine_type == "diffusion":
            from .diffusion_head import LatentDiffusionRefineHead
            self.refine_head: nn.Module | None = LatentDiffusionRefineHead(
                feat_dim=nf, num_axes=num_axes,
                t_inference=cfg.diffusion_t_inference,
            )
            self._refine_kind = "diffusion"
        elif cfg.refine_type == "adversarial":
            self.refine_head = AdversarialRefineHead(
                feat_dim=nf, num_axes=num_axes,
                hidden_dim=cfg.refine_hidden_dim or 128,
                n_blocks=cfg.refine_n_blocks or 8,
            )
            self._refine_kind = "adversarial"
        else:
            self.refine_head = None
            self._refine_kind = "none"
```

Then update the last block of `forward` to:

```python
        if self.refine_head is None:
            return coarse_rgb
        return self.refine_head(features, coarse_rgb, config)
```

(Both adversarial and diffusion heads have signature `(features, coarse_rgb, config) -> rgb`, so this works uniformly.)

Add a new method right after `forward`:

```python
    def forward_with_extras(
        self,
        rgb: torch.Tensor,
        clean_rgb: torch.Tensor,
        config: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Training-time forward exposing intermediate tensors.

        For diffusion refine head: returns (decoded_rgb, {"pred_latent", "target_latent"}).
        For other refine types: returns (forward(rgb, config), {}).
        """
        if self._refine_kind != "diffusion":
            return self.forward(rgb, config), {}

        lab_n = self.rgb_to_lab(rgb)
        task_vec = self.task_embed(config)
        x = self.stem(lab_n)
        skips: list[torch.Tensor] = []
        for stage, down in zip(self.enc_stages, self.downs):
            for blk in stage:
                x = blk(x, task_vec)
            skips.append(x)
            x = down(x)
        x = self.bottle_in(x)
        for blk in self.bottle:
            x = blk(x, task_vec)
        x = self.bottle_out(x)
        for up, proj, stage, skip in zip(self.ups, self.skip_proj, self.dec_stages, reversed(skips)):
            x = up(x)
            x = proj(torch.cat([x, skip], dim=1))
            for blk in stage:
                x = blk(x, task_vec)
        features = x
        delta_lab_n = self.head_lab_delta(x)
        ab_pred = self.head_ab_abs(x)
        lab_intermediate = lab_n + delta_lab_n
        w = config[:, 0:1].view(-1, 1, 1, 1)
        ab_out = w * ab_pred + (1.0 - w) * lab_intermediate[:, 1:3]
        L_out = lab_intermediate[:, 0:1]
        coarse_rgb = self.lab_to_rgb(torch.cat([L_out, ab_out], dim=1))

        pred_latent, target_latent, decoded_rgb = self.refine_head.forward_with_targets(
            features, coarse_rgb, clean_rgb, config)
        return decoded_rgb, {
            "pred_latent": pred_latent,
            "target_latent": target_latent,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nafnet_diffusion_wiring.py -v`
Expected: 4 passed.

- [ ] **Step 5: Verify no regressions**

Run: `uv run pytest tests/test_nafnet.py tests/test_adversarial_refine_head.py tests/test_legacy_checkpoint_load.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```sh
git add src/restora_models/models/nafnet.py tests/test_nafnet_diffusion_wiring.py
git commit -m "models: wire LatentDiffusionRefineHead into NAFNetMultiTask"
```

---

## Task 8: Trainer wiring for diffusion supervision

**Files:**
- Modify: `src/restora_models/train/trainer.py`
- Create: `tests/test_diffusion_trainer_smoke.py`

- [ ] **Step 1: Add a helper attribute in Trainer.__init__**

Edit `src/restora_models/train/trainer.py`. After the line `self.model = build_model(cfg.model, num_axes=len(AXES)).to(...)` in `__init__`, add:

```python
        self._is_diffusion_refine = (cfg.model.refine_type == "diffusion")
```

- [ ] **Step 2: Modify _train_step**

In `_train_step`, find the line `pred = self.model(degraded, config)` and the immediately following `ctx = LossContext(...)`. Replace those two with:

```python
            if self._is_diffusion_refine:
                pred, extras = self.model.forward_with_extras(degraded, clean, config)
            else:
                pred = self.model(degraded, config)
                extras = {}
            ctx = LossContext(pred_rgb=pred, clean_rgb=clean, degraded_rgb=degraded,
                              config=config, axes_active=axes, discriminator=self.disc,
                              pred_latent=extras.get("pred_latent"),
                              target_latent=extras.get("target_latent"))
```

- [ ] **Step 3: Modify _train_step_video**

In `_train_step_video`, find the line `pred_pair = self.model(deg_pair, cfg_pair)` and the immediately following `ctx = LossContext(...)`. Replace with:

```python
            if self._is_diffusion_refine:
                clean_pair_for_extras = torch.cat([clean_t, clean_tk], dim=0)
                pred_pair, extras = self.model.forward_with_extras(
                    deg_pair, clean_pair_for_extras, cfg_pair)
            else:
                pred_pair = self.model(deg_pair, cfg_pair)
                extras = {}
            pred_t, pred_tk = pred_pair[:B], pred_pair[B:]

            ctx = LossContext(
                pred_rgb=pred_t, clean_rgb=clean_t, degraded_rgb=deg_t,
                config=config, axes_active=axes, discriminator=self.disc,
                secondary_pred_rgb=pred_tk, flow_t_to_secondary=flow,
                pred_latent=extras.get("pred_latent"),
                target_latent=extras.get("target_latent"),
            )
```

(The downstream `clean_pair` variable used by `_disc_step` later in the same function is constructed independently; don't remove that.)

- [ ] **Step 4: Write the smoke test**

Create `tests/test_diffusion_trainer_smoke.py`:

```python
"""Smoke test: trainer.run_one_step() with refine_type='diffusion'."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from restora_models.config import (
    AugmentConfig, CompoundConfig, Config, DataConfig, ExportConfig, LoaderConfig,
    LossConfig, ModelConfig, OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
    VideoConfig,
)


@pytest.mark.skipif(
    not __import__('os').environ.get('REFINE_SLOW'),
    reason="downloads SD VAE; set REFINE_SLOW=1 to run",
)
def test_diffusion_trainer_runs_one_step(tmp_path):
    data_root = tmp_path / "imgs"
    data_root.mkdir()
    for i in range(8):
        img = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(data_root / f"img_{i}.jpg"), img)

    cfg = Config(
        run=RunConfig(name="diff-smoke", output_dir=str(tmp_path / "run")),
        model=ModelConfig(type="nafnet", size="tiny", refine_type="diffusion",
                          input_size=64, nf=8, enc_depths=[1,1,1,1],
                          bottle_blocks=1, hidden_dim=32),
        data=DataConfig(root=str(data_root), val_fraction=0.0,
                        loader=LoaderConfig(batch_size=2, num_workers=0)),
        compound=CompoundConfig(),
        losses=[LossConfig(name="l1_rgb", weight=1.0),
                LossConfig(name="l1_latent", weight=1.0)],
        optim_g=OptimConfig(lr=1e-4, fused=False),
        optim_d=OptimConfig(lr=1e-4, fused=False, weight_decay=0.0),
        scheduler=SchedulerConfig(total_steps=10, warmup_steps=2),
        train=TrainConfig(total_steps=10, amp="fp32", compile=False,
                          ema_decay=0.0, preview_every_s=0.0,
                          ckpt_every_steps=0, log_every_steps=10),
        export=ExportConfig(on_finish=False),
        video=VideoConfig(enabled=False),
    )
    from restora_models.train import Trainer
    t = Trainer(cfg)
    log = t.run_one_step()
    assert "total_g" in log
    assert torch.isfinite(torch.tensor(log["total_g"]))
```

- [ ] **Step 5: Run smoke test**

Run: `REFINE_SLOW=1 uv run pytest tests/test_diffusion_trainer_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 6: Run full suite**

Run: `uv run pytest -q`
Expected: 165+ passed.

- [ ] **Step 7: Commit**

```sh
git add src/restora_models/train/trainer.py tests/test_diffusion_trainer_smoke.py
git commit -m "trainer: wire pred_latent/target_latent into LossContext"
```

---

## Task 9: Stage 1 training config

**Files:**
- Create: `configs/b200-diffusion.yaml`
- Modify: `tests/test_configs_load.py`

- [ ] **Step 1: Write the failing test**

Edit `tests/test_configs_load.py`. Add at the bottom:

```python
def test_b200_diffusion_yaml_loads():
    cfg = load_config(ROOT / "b200-diffusion.yaml")
    assert cfg.model.refine_type == "diffusion"
    assert cfg.model.diffusion_t_inference == 0.2
    names = [l.name for l in cfg.losses]
    assert "l1_latent" in names
    assert "gan" not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_configs_load.py::test_b200_diffusion_yaml_loads -v`
Expected: FAIL.

- [ ] **Step 3: Write the config**

Create `configs/b200-diffusion.yaml`:

```yaml
# Stage 1 training config for the latent diffusion refine head.
#
# Prerequisite: configs/b200.yaml has finished (500k steps), final.pt
# is the starting point for the backbone + dual-output head.
#
# Launch:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#       uv run restora train --config configs/b200-diffusion.yaml \
#           --resume runs/<b200-run>/ckpt/final.pt --compile

defaults: large.yaml

run:
  name: "b200-diffusion-${date:%Y-%m-%d_%H-%M-%S}"

model:
  type: nafnet
  size: large
  input_size: 256
  task_embed_dim: 128
  refine_type: diffusion
  diffusion_t_inference: 0.2

data:
  root: "/workspace/data"
  val_fraction: 0.002
  num_fixed_preview_samples: 2
  num_random_preview_samples: 2
  loader:
    batch_size: 96
    num_workers: 16
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4

compound:
  identity_prob: 0.05
  axis_probs:
    colorize: 0.65
    denoise:  0.50
    sharpen:  0.50
    dejpeg:   0.50
    deblur:   0.50
  degradations:
    colorize: {}
    denoise:  { sigma_range: [0.005, 0.05] }
    sharpen:  { factor_choices: [2, 4, 8] }
    dejpeg:   { quality_range: [20, 70] }
    deblur:   { sigma_range: [1.0, 3.0], motion_prob: 0.2 }

# Diffusion subsumes GAN; chroma/colorfulness/freq weights bumped up to
# exploit the head's hallucinatory capacity on the hard axes.
losses:
  - {name: l1_latent,          weight: 1.0}
  - {name: l1_rgb,             weight: 0.5}
  - {name: perceptual_vgg16bn, weight: 0.5,  config: {criterion: l1}}
  - {name: chroma_lab,         weight: 0.20, apply_to_axes: [colorize]}
  - {name: colorfulness,       weight: 0.10, apply_to_axes: [colorize]}
  - {name: freq_l1,            weight: 0.40, apply_to_axes: [sharpen]}
  - {name: temporal_pair,      weight: 0.5}

optim_g:
  type: AdamW
  lr: 2.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.99]
  fused: true

optim_d:
  type: AdamW
  lr: 1.0e-4
  weight_decay: 0.0
  betas: [0.9, 0.99]
  fused: true

scheduler:
  type: cosine
  warmup_steps: 5000
  total_steps: 200000

train:
  total_steps: 200000
  amp: "bf16"
  memory_format: channels_last
  ema_decay: 0.999
  clip_grad_norm: 1.0
  preview_every_s: 120
  preview_history_every: 5000
  ckpt_every_steps: 10000
  ckpt_history_every: 20000
  log_every_steps: 50
  gan_warmup_start: 0
  gan_warmup_steps: 0

video:
  enabled: true
  root: "/workspace/data-videos"
  max_skip: 5
  hflip_prob: 0.5
  require_flow: true
  video_batch_prob: 0.20
  batch_size: 48
  num_workers: 8

export:
  on_finish: true
  opset: 17
  simplify: true
  dynamic_hw: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_configs_load.py::test_b200_diffusion_yaml_loads -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```sh
git add configs/b200-diffusion.yaml tests/test_configs_load.py
git commit -m "configs: b200-diffusion.yaml — Stage 1 training for diffusion head"
```

---

## Self-Review Notes (post-write check)

1. **Spec coverage:**
   - Frozen VAE wrapper → Task 2
   - LatentDiffusionRefineHead architecture → Tasks 3 + 4
   - AdaLN conditioning on config + timestep → Task 3
   - l1_latent loss → Task 5
   - Model integration → Tasks 6 + 7
   - Trainer integration → Task 8
   - Stage 1 config → Task 9
   - ONNX export — deferred to a follow-up plan (intentional)
   - Backbone freezing for Stage 1 — deferred (Stage 1 trains everything; backbone has a small fine-tune gradient. Can add a freeze flag in a follow-up commit if Stage 1 results show easy-axis regression.)

2. **Placeholder scan:** none.

3. **Type consistency:** `forward_with_targets` returns `(pred_z_clean, target_z_clean, decoded_rgb)` in Task 4, consumed correctly in Task 7's `forward_with_extras`. `LossContext.pred_latent` / `target_latent` field names consistent across Tasks 5, 7, 8.

4. **Open follow-up (Task 10, separate plan):** backbone freezing flag + ONNX export wrapper extension. Both wait until Stage 1 trains and validates.

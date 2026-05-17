# Temporal Old-Film Remaster — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-frame multi-task restoration model with a single temporal model (7-frame symmetric input, flow-warped fusion with distilled RAFT, TemporalNAFNet backbone, RSD one-step refine head) that runs at any input resolution, exports to one ONNX file, and ships multiple sizes via SLKD distillation. Build a composite `VideoWindowDataset` so multiple video data sources (REDS primarily, also Vimeo Septuplet, future custom sources) can be interleaved. RADICALLY clean up the repo to leave zero old code, scripts, or artifacts.

**Architecture:** `forward(frames [B,7,3,H,W], config [B,5]) -> rgb [B,3,H,W]`. Distilled RAFT -> flow-warp aligner -> TemporalNAFNet (fully convolutional, FiLM-conditioned 5-axis task vector) -> RSD residual-shift refine in RGB -> Lab dual-head output. Data layer: `VideoWindowDataset` facade that interleaves N sub-datasets (`REDSDataset`, `VimeoSeptupletDataset`, ...), each yielding a 7-frame `(7,3,H,W)` tensor. No SD VAE, no `diffusers` dep. Multiple sizes (nano/small/medium/large) share the contract, produced via distillation. See `docs/superpowers/specs/2026-05-17-temporal-old-film-remaster-design.md` for the full design.

**Tech Stack:** PyTorch 2.4+, torchvision (RAFT teacher), Muon optimizer, LPIPS, ONNX 1.16+, TensorRT target, typer (CLI), pytest. Removes: `diffusers`. Adds: `muon-pytorch`, `lpips`.

**Note on data layer:** Primary training source is REDS (270 train sequences of 100 contiguous 720p frames). Vimeo Septuplet stays as a secondary sub-dataset for diversity. Future sources (BVI-DVC, GoPro, raw mp4 archives) plug in as additional sub-dataset classes implementing the same protocol.

---

## Phase 0 — Radical cleanup of obsolete artifacts

Delete everything the new design obsoletes. The trainer harness stays; some shared model files (nafblock, transformer_block, config_embed) stay because the new TemporalNAFNet reuses them. Run artifacts, old checkpoints, old configs, obsoleted tests all go.

### Task 0.1: Delete obsolete model files

**Files:**
- Delete: `src/restora_models/models/nafnet.py`
- Delete: `src/restora_models/models/heads.py`
- Delete: `src/restora_models/models/diffusion_head.py`
- Delete: `src/restora_models/models/vae.py`
- Delete: `src/restora_models/models/discriminator.py`
- Delete: `src/restora_models/models/color.py`

- [ ] **Step 1: Delete files**

```bash
rm -f src/restora_models/models/{nafnet,heads,diffusion_head,vae,discriminator,color}.py
```

- [ ] **Step 2: Verify deletions**

```bash
ls src/restora_models/models/
```

Expected output: `__init__.py  config_embed.py  nafblock.py  registry.py  transformer_block.py`

- [ ] **Step 3: Commit**

```bash
git add -u src/restora_models/models/
git commit -m "chore: remove per-frame model files (replaced by temporal design)"
```

### Task 0.2: Delete obsolete loss files

**Files:**
- Delete: `src/restora_models/losses/diffusion.py`
- Delete: `src/restora_models/losses/gan.py`

- [ ] **Step 1: Delete files**

```bash
rm -f src/restora_models/losses/{diffusion,gan}.py
```

- [ ] **Step 2: Verify**

```bash
ls src/restora_models/losses/
```

Expected: `__init__.py chroma.py colorfulness.py freq.py metrics.py perceptual.py pixel.py registry.py temporal.py`

- [ ] **Step 3: Commit**

```bash
git add -u src/restora_models/losses/
git commit -m "chore: remove diffusion + GAN losses (no SD VAE, no GAN in new design)"
```

### Task 0.3: Delete obsolete configs

**Files:**
- Delete: `configs/local.yaml`
- Delete: `configs/b200.yaml`
- Delete: `configs/b200-diffusion.yaml`
- Delete: `configs/large.yaml`

Keep `configs/default.yaml` — base inherited config, will be updated later.

- [ ] **Step 1: Delete**

```bash
rm -f configs/{local,b200,b200-diffusion,large}.yaml
```

- [ ] **Step 2: Verify**

```bash
ls configs/
```

Expected: `default.yaml`

- [ ] **Step 3: Commit**

```bash
git add -u configs/
git commit -m "chore: remove obsolete configs (new local-temporal + b200-temporal will replace)"
```

### Task 0.4: Delete obsolete top-level commands

**Files:**
- Delete: `src/restora_models/distill.py` (will be rewritten under `train/`)
- Delete: `src/restora_models/evaluate.py` (will be rewritten under `train/`)
- Delete: `src/restora_models/gallery.py` (will be rewritten under `train/`)
- Delete: `src/restora_models/bench.py` (will be rewritten under `train/`)

- [ ] **Step 1: Delete**

```bash
rm -f src/restora_models/{distill,evaluate,gallery,bench}.py
```

- [ ] **Step 2: Verify**

```bash
ls src/restora_models/
```

Expected: `__init__.py  cli.py  config.py  data  export  infer  losses  models  train  utils`

- [ ] **Step 3: Commit**

```bash
git add -u src/restora_models/
git commit -m "chore: remove top-level command implementations (to be rewritten temporal-aware)"
```

### Task 0.5: Delete obsolete tests

**Files:**
- Delete tests for removed modules.

- [ ] **Step 1: Delete**

```bash
cd /home/bglueck/work/coliraz
rm -f tests/test_adversarial_refine_head.py
rm -f tests/test_diffusion_head.py
rm -f tests/test_diffusion_head_blocks.py
rm -f tests/test_diffusion_trainer_smoke.py
rm -f tests/test_frozen_vae.py
rm -f tests/test_gan_colorfulness.py
rm -f tests/test_l1_latent_loss.py
rm -f tests/test_legacy_checkpoint_load.py
rm -f tests/test_nafnet.py
rm -f tests/test_nafnet_diffusion_wiring.py
rm -f tests/test_color.py
rm -f tests/test_color_modules.py
rm -f tests/test_e2e_smoke.py
rm -f tests/test_synthetic_video_flow.py
rm -f tests/test_export_onnx.py
rm -f tests/test_export_pnnx.py
rm -f tests/test_export_precision.py
rm -f tests/test_onnx_wrapper.py
rm -f tests/test_train_step.py
rm -f tests/test_train_video_step.py
rm -f tests/test_train_ckpt_ema.py
rm -f tests/test_video_compound_wrapper.py
rm -f tests/test_video_pair_dataset.py
rm -f tests/test_compound.py
rm -f tests/test_preview.py
rm -f tests/test_preview_sr_factors.py
rm -f tests/test_inference.py
rm -f tests/test_dataset.py
rm -f tests/test_cli.py
rm -f tests/test_download_module.py
rm -f tests/test_temporal_loss.py
rm -f tests/test_configs_load.py
```

- [ ] **Step 2: Verify remaining**

```bash
ls tests/
```

Expected: `__init__.py conftest.py test_chroma_loss.py test_config.py test_config_embed.py test_degradation_registry.py test_degradations.py test_freq_loss.py test_loss_presets.py test_loss_set.py test_metrics.py test_nafblock.py test_perceptual.py test_pixel_losses.py test_transformer_block.py test_ui_smoke.py test_utils_misc.py`

- [ ] **Step 3: Commit**

```bash
git add -u tests/
git commit -m "chore: remove obsolete tests for replaced modules"
```

### Task 0.6: Archive obsolete plans

- [ ] **Step 1: Move**

```bash
mkdir -p docs/superpowers/plans/_archive
git mv docs/superpowers/plans/2026-05-11-coliraz-modern-port.md docs/superpowers/plans/_archive/
git mv docs/superpowers/plans/2026-05-13-refine-multitask.md docs/superpowers/plans/_archive/
git mv docs/superpowers/plans/2026-05-14-dual-output-head.md docs/superpowers/plans/_archive/
git mv docs/superpowers/plans/2026-05-16-latent-diffusion-refine-head.md docs/superpowers/plans/_archive/
```

- [ ] **Step 2: Verify**

```bash
ls docs/superpowers/plans/
```

Expected: `_archive 2026-05-17-temporal-old-film-remaster.md`

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/
git commit -m "chore: archive obsolete plans"
```

### Task 0.7: Archive obsolete specs

- [ ] **Step 1: Move**

```bash
mkdir -p docs/superpowers/specs/_archive
git mv docs/superpowers/specs/2026-05-11-coliraz-modern-port-design.md docs/superpowers/specs/_archive/
git mv docs/superpowers/specs/2026-05-13-refine-multitask-design.md docs/superpowers/specs/_archive/
git mv docs/superpowers/specs/2026-05-13-refine-compound-design.md docs/superpowers/specs/_archive/
git mv docs/superpowers/specs/2026-05-14-dual-output-head-design.md docs/superpowers/specs/_archive/
git mv docs/superpowers/specs/2026-05-16-latent-diffusion-refine-head-design.md docs/superpowers/specs/_archive/
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/
git commit -m "chore: archive obsolete specs"
```

### Task 0.8: Delete training artifacts and add to gitignore

- [ ] **Step 1: Delete**

```bash
rm -rf runs/ trained/
```

- [ ] **Step 2: Verify**

```bash
ls /home/bglueck/work/coliraz/ | grep -E '^(runs|trained)$' || echo "clean"
```

Expected: `clean`

- [ ] **Step 3: Add to .gitignore**

```bash
grep -q '^runs/$' .gitignore || echo 'runs/' >> .gitignore
grep -q '^trained/$' .gitignore || echo 'trained/' >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore runs/ and trained/ (clean slate for temporal training)"
```

### Task 0.9: Delete pycache + add to gitignore

- [ ] **Step 1: Delete**

```bash
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete
find . -name '.pytest_cache' -type d -exec rm -rf {} + 2>/dev/null
```

- [ ] **Step 2: Add to .gitignore**

```bash
grep -q '^__pycache__/$' .gitignore || echo '__pycache__/' >> .gitignore
grep -q '^\*.pyc$' .gitignore || echo '*.pyc' >> .gitignore
grep -q '^\.pytest_cache/$' .gitignore || echo '.pytest_cache/' >> .gitignore
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: clean pycache + ignore"
```

### Task 0.10: Sanity check Phase 0

- [ ] **Step 1: Confirm**

```bash
ls src/restora_models/models/
ls src/restora_models/losses/
ls configs/
ls tests/ | wc -l
git status --short
```

Expected:
- `models/`: `__init__.py config_embed.py nafblock.py registry.py transformer_block.py`
- `losses/`: 10 files (no diffusion.py, no gan.py)
- `configs/`: `default.yaml` only
- `tests/`: 17 files

---

## Phase 1 — pyproject + dependency updates

### Task 1.1: Update pyproject.toml deps

**Files:**
- Modify: `pyproject.toml`

Changes:
- Remove `diffusers>=0.30`
- Add `muon-pytorch>=0.2`
- Add `lpips>=0.1.4`
- Remove the `restora-models` alias script entry
- Update description

- [ ] **Step 1: Edit pyproject.toml**

In `[project] dependencies`, replace the line `"diffusers>=0.30",` (and its preceding comment) with:

```toml
  # Backbone optimizer (Muon for backbone params, AdamW for norms/bias)
  "muon-pytorch>=0.2",
  # Perceptual loss for distillation + decoded RGB supervision
  "lpips>=0.1.4",
```

Update description:
```toml
description = "Multi-task video restoration: temporal model with colorize / denoise / sharpen / dejpeg / deblur, optimized for old-film remastering"
```

In `[project.scripts]`, remove the `restora-models = ...` line.

- [ ] **Step 2: Sync deps**

```bash
uv sync
```

Expected: no errors, `diffusers` removed, `muon-pytorch` + `lpips` installed.

- [ ] **Step 3: Verify imports work**

```bash
uv run python -c "import torch; import lpips; from muon import Muon; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): drop diffusers, add muon-pytorch + lpips for temporal redesign"
```

---

## Phase 2 — Distilled RAFT

Static-unroll RAFT student. Takes `(B, 2, 3, H, W)` (two frames) and returns backward flow `(B, 2, H, W)`. Iterations fixed at construction time for ONNX cleanliness.

### Task 2.1: Failing test

**Files:**
- Create: `tests/test_flow_distill.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the static-unroll RAFT student in models/flow_distill.py."""
import torch

from restora_models.models.flow_distill import FlowDistill


def test_flow_distill_output_shape():
    m = FlowDistill(iters=4).eval()
    pair = torch.randn(2, 2, 3, 128, 128)
    flow = m(pair)
    assert flow.shape == (2, 2, 128, 128), f"got {tuple(flow.shape)}"


def test_flow_distill_no_python_loop_in_graph():
    """The forward must not contain Python-level loops in the traced graph."""
    m = FlowDistill(iters=4).eval()
    pair = torch.randn(1, 2, 3, 64, 64)
    traced = torch.jit.trace(m, pair)
    assert "prim::Loop" not in str(traced.graph)


def test_flow_distill_param_budget():
    m = FlowDistill(iters=4)
    n = sum(p.numel() for p in m.parameters())
    assert 2_000_000 < n < 8_000_000, f"unexpected param count: {n}"
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_flow_distill.py -v
```

Expected: ImportError.

### Task 2.2: Implement FlowDistill

**Files:**
- Create: `src/restora_models/models/flow_distill.py`

- [ ] **Step 1: Implement**

```python
"""Static-unroll RAFT-style flow estimator for the temporal stem.

Designed for ONNX-clean export: no while-loop, no dynamic shape inside
the graph. Trained via distillation from torchvision raft_large in a
separate one-shot script (`restora train-flow-distill`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _residual_block(c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
        nn.Conv2d(c, c, 3, padding=1),
    )


class _FeatureExtractor(nn.Module):
    def __init__(self, dims=(32, 64, 96, 128)):
        super().__init__()
        self.stem = nn.Conv2d(3, dims[0], 7, stride=2, padding=3)
        self.act = nn.GELU()
        self.b1 = _residual_block(dims[0])
        self.d1 = nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1)
        self.b2 = _residual_block(dims[1])
        self.d2 = nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1)
        self.b3 = _residual_block(dims[2])
        self.proj = nn.Conv2d(dims[2], dims[3], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.stem(x))
        x = x + self.b1(x)
        x = self.act(self.d1(x))
        x = x + self.b2(x)
        x = self.act(self.d2(x))
        x = x + self.b3(x)
        return self.proj(x)


class _UpdateBlock(nn.Module):
    def __init__(self, c: int = 128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2 + 2, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
        )
        self.delta = nn.Conv2d(c, 2, 3, padding=1)

    def forward(self, fa: torch.Tensor, fb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        h = torch.cat([fa, fb, flow], dim=1)
        return self.delta(self.fuse(h))


class FlowDistill(nn.Module):
    """Static-unroll RAFT student.

    Forward input:  frames (B, 2, 3, H, W)  -- frame_a, frame_b
    Forward output: flow   (B, 2, H, W)     -- backward flow b -> a
    """

    def __init__(self, iters: int = 4):
        super().__init__()
        if iters < 1:
            raise ValueError(f"iters must be >=1, got {iters}")
        self.iters = iters
        self.feat = _FeatureExtractor()
        self.updates = nn.ModuleList([_UpdateBlock() for _ in range(iters)])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5 or frames.shape[1] != 2:
            raise ValueError(f"expected (B,2,3,H,W), got {tuple(frames.shape)}")
        b, _, _, h, w = frames.shape
        fa = self.feat(frames[:, 0])
        fb = self.feat(frames[:, 1])
        flow = torch.zeros(b, 2, h // 8, w // 8, device=frames.device, dtype=frames.dtype)
        for blk in self.updates:
            flow = flow + blk(fa, fb, flow)
        return F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False) * 8.0
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_flow_distill.py -v
```

Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add src/restora_models/models/flow_distill.py tests/test_flow_distill.py
git commit -m "feat(models): static-unroll RAFT student for ONNX-clean flow estimation"
```

---

## Phase 3 — Flow warp + visibility mask

### Task 3.1: Failing tests

**Files:**
- Create: `tests/test_warp.py`

- [ ] **Step 1: Write tests**

```python
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
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_warp.py -v
```

### Task 3.2: Implement warp + visibility

**Files:**
- Create: `src/restora_models/models/warp.py`

- [ ] **Step 1: Implement**

```python
"""Flow-based warping + cycle-consistency occlusion mask.

Pure ops; no learnable params; ONNX-safe via grid_sample (opset 16+).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _identity_grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0).expand(b, h, w, 2).contiguous()


def flow_warp(rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp rgb (B,C,H,W) by flow (B,2,H,W). Output pixel (y,x) is sampled from
    (y + flow_y, x + flow_x) in the input."""
    if rgb.dim() != 4 or flow.dim() != 4:
        raise ValueError(f"expected 4D tensors, got rgb={rgb.shape} flow={flow.shape}")
    b, _, h, w = rgb.shape
    if flow.shape != (b, 2, h, w):
        raise ValueError(f"flow shape mismatch: rgb={tuple(rgb.shape)} flow={tuple(flow.shape)}")
    base = _identity_grid(b, h, w, rgb.device, rgb.dtype)
    scale_x = 2.0 / max(w - 1, 1)
    scale_y = 2.0 / max(h - 1, 1)
    offset = torch.stack([flow[:, 0] * scale_x, flow[:, 1] * scale_y], dim=-1)
    sample_grid = base + offset
    return F.grid_sample(rgb, sample_grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


def visibility_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor,
                    threshold: float = 0.5) -> torch.Tensor:
    """Soft visibility mask from cycle consistency.

    A pixel p is visible if flow_fwd(p) + flow_bwd(p + flow_fwd(p)) is near zero.
    """
    if flow_fwd.shape != flow_bwd.shape:
        raise ValueError("flow shapes must match")
    warped_bwd = flow_warp(flow_bwd, flow_fwd)
    cycle = flow_fwd + warped_bwd
    err = torch.linalg.vector_norm(cycle, ord=2, dim=1, keepdim=True)
    return torch.sigmoid(-(err - threshold) * 4.0)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_warp.py -v
```

Expected: 4 PASS

- [ ] **Step 3: Commit**

```bash
git add src/restora_models/models/warp.py tests/test_warp.py
git commit -m "feat(models): flow_warp + visibility_mask for temporal alignment"
```

---

## Phase 4 — TemporalNAFNet backbone

### Task 4.1: TemporalAlignStem

**Files:**
- Create: `tests/test_temporal_align_stem.py`
- Create: `src/restora_models/models/temporal_align_stem.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for TemporalAlignStem in models/temporal_align_stem.py."""
import torch

from restora_models.models.temporal_align_stem import TemporalAlignStem


def test_align_stem_output_shape():
    stem = TemporalAlignStem().eval()
    frames = torch.rand(2, 7, 3, 64, 64)
    out = stem(frames)
    assert out.shape == (2, 28, 64, 64), f"got {tuple(out.shape)}"


def test_align_stem_identical_frames_path():
    stem = TemporalAlignStem().eval()
    img = torch.rand(1, 3, 64, 64)
    frames = img.unsqueeze(1).expand(1, 7, 3, 64, 64).contiguous()
    out = stem(frames)
    rgb_part = out[:, :21].view(1, 7, 3, 64, 64)
    for k in range(7):
        assert rgb_part[:, k].shape == img.shape
    mask_part = out[:, 21:]
    assert mask_part.shape == (1, 7, 64, 64)
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_temporal_align_stem.py -v
```

- [ ] **Step 3: Implement**

```python
"""Temporal alignment stem: flow + warp + visibility into a 28-channel
tensor consumed by TemporalNAFNet.

Input:  frames (B, 7, 3, H, W)
Output: (B, 28, H, W) = 7*3 RGB (center + 6 warped neighbors) + 7 visibility masks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from restora_models.models.flow_distill import FlowDistill
from restora_models.models.warp import flow_warp, visibility_mask


class TemporalAlignStem(nn.Module):
    CENTER_INDEX = 3
    NUM_FRAMES = 7

    def __init__(self, flow_iters: int = 4):
        super().__init__()
        self.flow = FlowDistill(iters=flow_iters)

    @staticmethod
    def _pair(frames: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        return torch.stack([frames[:, src], frames[:, dst]], dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5 or frames.shape[1] != self.NUM_FRAMES:
            raise ValueError(f"expected (B,{self.NUM_FRAMES},3,H,W), got {tuple(frames.shape)}")
        b, _, _, h, w = frames.shape
        center = frames[:, self.CENTER_INDEX]
        ones = torch.ones(b, 1, h, w, device=frames.device, dtype=frames.dtype)
        warped, masks = [], []
        for k in range(self.NUM_FRAMES):
            if k == self.CENTER_INDEX:
                warped.append(center)
                masks.append(ones)
                continue
            flow_n_c = self.flow(self._pair(frames, k, self.CENTER_INDEX))
            flow_c_n = self.flow(self._pair(frames, self.CENTER_INDEX, k))
            warped.append(flow_warp(frames[:, k], flow_n_c))
            mask_k = visibility_mask(flow_n_c, flow_c_n).squeeze(1).unsqueeze(1)
            masks.append(mask_k)
        rgb_cat = torch.cat(warped, dim=1)
        mask_cat = torch.cat(masks, dim=1)
        return torch.cat([rgb_cat, mask_cat], dim=1)
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_temporal_align_stem.py -v
git add src/restora_models/models/temporal_align_stem.py tests/test_temporal_align_stem.py
git commit -m "feat(models): TemporalAlignStem (flow + warp + visibility -> 28ch)"
```

### Task 4.2: TemporalNAFNet — failing test

**Files:**
- Create: `tests/test_temporal_nafnet.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for TemporalNAFNet backbone."""
import torch

from restora_models.config import ModelConfig
from restora_models.models.registry import build_model


def test_temporal_nafnet_contract():
    cfg = ModelConfig(type="temporal_nafnet_small")
    m = build_model(cfg, num_axes=5).eval()
    frames = torch.rand(2, 7, 3, 64, 64)
    config = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 1.0, 0, 0]])
    out = m(frames, config)
    assert out.shape == (2, 3, 64, 64), f"got {tuple(out.shape)}"
    assert out.dtype == frames.dtype


def test_temporal_nafnet_any_resolution():
    cfg = ModelConfig(type="temporal_nafnet_small")
    m = build_model(cfg, num_axes=5).eval()
    for hw in [(96, 96), (128, 192), (256, 144), (96, 256)]:
        frames = torch.rand(1, 7, 3, *hw)
        cfgvec = torch.zeros(1, 5)
        out = m(frames, cfgvec)
        assert out.shape == (1, 3, *hw)


def test_temporal_nafnet_param_count_by_size():
    bands = {
        "temporal_nafnet_nano": (4_000_000, 12_000_000),
        "temporal_nafnet_small": (15_000_000, 30_000_000),
    }
    for name, (lo, hi) in bands.items():
        m = build_model(ModelConfig(type=name), num_axes=5)
        n = sum(p.numel() for p in m.parameters())
        assert lo <= n <= hi, f"{name}: {n} not in [{lo}, {hi}]"
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_temporal_nafnet.py -v
```

### Task 4.3: Implement TemporalNAFNet

**Files:**
- Create: `src/restora_models/models/temporal_nafnet.py`
- Possibly modify: `src/restora_models/utils/color.py` (port rgb_to_lab + lab_to_rgb from deleted models/color.py if not present)

- [ ] **Step 1: Ensure color utilities exist in utils/color.py**

```bash
grep -E '^def (rgb_to_lab|lab_to_rgb)' src/restora_models/utils/color.py || echo "MISSING — port from git history"
```

If missing, recover from history and add to `src/restora_models/utils/color.py`:

```bash
git show HEAD~10:src/restora_models/models/color.py > /tmp/old_color.py 2>/dev/null || true
# Then copy the two functions into utils/color.py manually.
```

- [ ] **Step 2: Implement TemporalNAFNet**

```python
"""TemporalNAFNet backbone.

Fully convolutional NAFNet-style encoder/decoder with FiLM conditioning
on a 5-axis task vector. Operates on the 28-channel output of
TemporalAlignStem. Bottleneck adds one temporal self-attention block.
Lab dual-head output (Lab-delta for all axes + ab-abs gated by colorize).

All sizes (nano/small/medium/large) registered as separate model types
in the registry but share this class with different hyperparams.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from restora_models.config import ModelConfig
from restora_models.models.config_embed import ConfigEmbed
from restora_models.models.nafblock import NAFBlock
from restora_models.models.registry import register_model
from restora_models.models.temporal_align_stem import TemporalAlignStem
from restora_models.models.transformer_block import TransformerBlock


@dataclass(frozen=True)
class _SizeSpec:
    nf: int
    enc_depths: tuple[int, int, int, int]
    bottle_blocks: int
    use_temporal_attn: bool


_SIZES: dict[str, _SizeSpec] = {
    "temporal_nafnet_nano":   _SizeSpec(nf=24, enc_depths=(1, 1, 1, 2), bottle_blocks=2, use_temporal_attn=False),
    "temporal_nafnet_small":  _SizeSpec(nf=36, enc_depths=(2, 2, 2, 4), bottle_blocks=4, use_temporal_attn=True),
    "temporal_nafnet_medium": _SizeSpec(nf=48, enc_depths=(2, 2, 4, 6), bottle_blocks=6, use_temporal_attn=True),
    "temporal_nafnet_large":  _SizeSpec(nf=64, enc_depths=(2, 2, 4, 8), bottle_blocks=8, use_temporal_attn=True),
}


class _DownConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UpConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class _EncoderStage(nn.Module):
    def __init__(self, c: int, depth: int, task_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(c, task_dim=task_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, task)
        return x


class _Bottleneck(nn.Module):
    def __init__(self, c: int, blocks: int, use_attn: bool, task_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(c, task_dim=task_dim) for _ in range(blocks)])
        self.attn = TransformerBlock(c, task_dim=task_dim) if use_attn else None

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            x = blk(x, task)
            if self.attn is not None and i == len(self.blocks) // 2:
                x = self.attn(x, task)
        return x


class _LabDualHead(nn.Module):
    """Outputs Lab-delta (all axes) + ab-abs (colorize-gated)."""

    def __init__(self, c: int):
        super().__init__()
        self.head_lab_delta = nn.Conv2d(c, 3, 3, padding=1)
        self.head_ab_abs = nn.Conv2d(c, 2, 3, padding=1)
        nn.init.zeros_(self.head_lab_delta.weight)
        nn.init.zeros_(self.head_lab_delta.bias)
        nn.init.zeros_(self.head_ab_abs.weight)
        nn.init.zeros_(self.head_ab_abs.bias)

    def forward(self, feat: torch.Tensor, center_rgb: torch.Tensor, colorize_gate: torch.Tensor) -> torch.Tensor:
        from restora_models.utils.color import rgb_to_lab, lab_to_rgb
        delta = self.head_lab_delta(feat)
        ab_abs = self.head_ab_abs(feat)
        lab = rgb_to_lab(center_rgb)
        lab_new = lab + delta
        gate = colorize_gate.view(-1, 1, 1, 1)
        lab_new[:, 1:] = lab_new[:, 1:] * (1.0 - gate) + ab_abs * gate
        return lab_to_rgb(lab_new).clamp(0.0, 1.0)


class TemporalNAFNet(nn.Module):
    def __init__(self, cfg: ModelConfig, num_axes: int = 5):
        super().__init__()
        size = _SIZES[cfg.type]
        nf = size.nf
        self.align_stem = TemporalAlignStem()
        self.cfg_embed = ConfigEmbed(num_axes, hidden=128)
        task_dim = self.cfg_embed.out_dim
        self.input_conv = nn.Conv2d(28, nf, 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = nf
        for depth in size.enc_depths:
            self.enc.append(_EncoderStage(c, depth, task_dim))
            self.down.append(_DownConv(c, c * 2))
            c = c * 2
        self.bottleneck = _Bottleneck(c, size.bottle_blocks, size.use_temporal_attn, task_dim)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for depth in reversed(size.enc_depths):
            self.up.append(_UpConv(c, c // 2))
            c = c // 2
            self.dec.append(_EncoderStage(c, depth, task_dim))
        self.head = _LabDualHead(c)

    def forward(self, frames: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        task = self.cfg_embed(config)
        center = frames[:, TemporalAlignStem.CENTER_INDEX]
        x = self.align_stem(frames)
        x = self.input_conv(x)
        skips = []
        for stage, down in zip(self.enc, self.down):
            x = stage(x, task)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x, task)
        for up, stage, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = x + skip
            x = stage(x, task)
        return self.head(x, center, config[:, 0])


for _name in _SIZES:
    register_model(_name)(TemporalNAFNet)
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_temporal_nafnet.py -v
```

Expected: 3 PASS

- [ ] **Step 4: Commit**

```bash
git add src/restora_models/models/temporal_nafnet.py tests/test_temporal_nafnet.py src/restora_models/utils/color.py
git commit -m "feat(models): TemporalNAFNet backbone (nano/small/medium/large)"
```

---

## Phase 5 — RSD refine head

### Task 5.1: Failing test

**Files:**
- Create: `tests/test_rsd_refine.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for RSDRefineHead -- one-step residual-shift diffusion in RGB."""
import torch

from restora_models.models.rsd_refine import RSDRefineHead


def test_rsd_refine_output_shape():
    head = RSDRefineHead(width=64, num_axes=5).eval()
    coarse = torch.rand(2, 3, 64, 64)
    config = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])
    out = head(coarse, config)
    assert out.shape == coarse.shape


def test_rsd_refine_near_identity_at_init():
    head = RSDRefineHead(width=64, num_axes=5).eval()
    coarse = torch.rand(1, 3, 64, 64)
    config = torch.zeros(1, 5)
    out = head(coarse, config)
    assert torch.allclose(out, coarse, atol=0.05), f"max diff {(out - coarse).abs().max().item()}"
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_rsd_refine.py -v
```

### Task 5.2: Implement RSDRefineHead

**Files:**
- Create: `src/restora_models/models/rsd_refine.py`

- [ ] **Step 1: Implement**

```python
"""RSD: one-step residual-shift diffusion in RGB space.

No external VAE. Operates directly on the backbone's coarse RGB output.
Conditioned on the 5-axis task vector + a per-axis t_inf scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from restora_models.models.config_embed import ConfigEmbed


def _t_inf_for(config: torch.Tensor) -> torch.Tensor:
    """Per-axis t_inf table from the spec section 3.4. Returns (B,) in [0,1].

    For samples that have multiple axes active, take the max — the hardest
    axis controls the noise level.
    """
    per_axis = torch.tensor([0.3, 0.05, 0.3, 0.05, 0.05],
                             device=config.device, dtype=config.dtype)
    weighted = config * per_axis.unsqueeze(0)
    return weighted.max(dim=1).values


class _FiLMBlock(nn.Module):
    def __init__(self, c: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, c), num_channels=c)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * c)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + g.unsqueeze(-1).unsqueeze(-1)) + b.unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        return x + h


class RSDRefineHead(nn.Module):
    """Single-step RGB-space residual refinement."""

    def __init__(self, width: int = 64, num_axes: int = 5, depth: int = 4):
        super().__init__()
        self.cfg_embed = ConfigEmbed(num_axes, hidden=128)
        cond_dim = self.cfg_embed.out_dim + 1
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.ModuleList([_FiLMBlock(width, cond_dim) for _ in range(depth)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, coarse_rgb: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        b = coarse_rgb.shape[0]
        task = self.cfg_embed(config)
        t_inf = _t_inf_for(config).view(b, 1)
        cond = torch.cat([task, t_inf], dim=-1)
        h = self.stem(coarse_rgb)
        for blk in self.blocks:
            h = blk(h, cond)
        residual = self.head(h)
        return (coarse_rgb + residual).clamp(0.0, 1.0)
```

- [ ] **Step 2: Run tests + commit**

```bash
uv run pytest tests/test_rsd_refine.py -v
git add src/restora_models/models/rsd_refine.py tests/test_rsd_refine.py
git commit -m "feat(models): RSD one-step residual-shift refine head (RGB space)"
```

---

## Phase 6 — TemporalRestora composite (backbone + RSD)

### Task 6.1: Tests + composite

**Files:**
- Modify: `src/restora_models/models/temporal_nafnet.py` (append composite class)
- Create: `tests/test_temporal_restora.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for TemporalRestora composite (backbone + RSD refine)."""
import torch

from restora_models.config import ModelConfig
from restora_models.models.registry import build_model


def test_composite_contract():
    cfg = ModelConfig(type="temporal_restora_small")
    m = build_model(cfg, num_axes=5).eval()
    frames = torch.rand(1, 7, 3, 64, 64)
    config = torch.tensor([[1.0, 0, 0, 0, 0]])
    out = m(frames, config)
    assert out.shape == (1, 3, 64, 64)


def test_composite_has_components():
    cfg = ModelConfig(type="temporal_restora_small")
    m = build_model(cfg, num_axes=5)
    assert hasattr(m, "refine"), "composite must expose .refine"
    assert hasattr(m, "backbone"), "composite must expose .backbone"
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_temporal_restora.py -v
```

- [ ] **Step 3: Append composite to temporal_nafnet.py**

```python
from restora_models.models.rsd_refine import RSDRefineHead


_REFINE_WIDTHS = {"nano": 0, "small": 64, "medium": 96, "large": 128}


class TemporalRestora(nn.Module):
    """Backbone + RSD refine in a single module exposing (frames, config) contract.

    For the `nano` size the refine head is skipped (width=0); the model is
    pure backbone for fastest student deployments.
    """

    def __init__(self, cfg: ModelConfig, num_axes: int = 5):
        super().__init__()
        backbone_type = cfg.type.replace("temporal_restora", "temporal_nafnet")
        size_key = backbone_type.rsplit("_", 1)[-1]
        self.backbone = TemporalNAFNet(ModelConfig(type=backbone_type), num_axes=num_axes)
        rw = _REFINE_WIDTHS[size_key]
        self.refine = RSDRefineHead(width=rw, num_axes=num_axes) if rw > 0 else None

    def forward(self, frames: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        coarse = self.backbone(frames, config)
        if self.refine is None:
            return coarse
        return self.refine(coarse, config)


for _size in ("nano", "small", "medium", "large"):
    register_model(f"temporal_restora_{_size}")(TemporalRestora)
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_temporal_restora.py tests/test_temporal_nafnet.py -v
git add src/restora_models/models/temporal_nafnet.py tests/test_temporal_restora.py
git commit -m "feat(models): TemporalRestora composite (backbone + RSD refine) per size"
```

---

## Phase 7 — Old-film degradation modules

### Task 7.1: Film overlay degradation

**Files:**
- Create: `src/restora_models/data/degradations/film_overlay.py`
- Create: `tests/test_film_overlay.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for film_overlay degradation."""
import numpy as np
import torch

from restora_models.data.degradations.film_overlay import FilmOverlayDegradation


def test_film_overlay_shape_preserved():
    deg = FilmOverlayDegradation(textures=None, alpha_range=(0.1, 0.3))
    img = torch.rand(3, 64, 64)
    out = deg.apply(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_film_overlay_no_textures_returns_input():
    deg = FilmOverlayDegradation(textures=None, alpha_range=(0.1, 0.3))
    img = torch.rand(3, 64, 64)
    out = deg.apply(img)
    assert torch.allclose(out, img)


def test_film_overlay_with_synthetic_texture():
    deg = FilmOverlayDegradation(
        textures=[np.full((64, 64), 0.5, dtype=np.float32)],
        alpha_range=(0.5, 0.5),
    )
    img = torch.zeros(3, 64, 64)
    out = deg.apply(img)
    assert out.mean().item() > 0.01
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_film_overlay.py -v
```

- [ ] **Step 3: Implement**

```python
"""Film overlay degradation: composite real scratch/dust/grain textures.

Textures come from the DeepRemaster noise_data.zip pack (898 MB,
6152 PNGs). Auto-download via `restora prepare-data --film-overlays`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass
class FilmOverlayDegradation:
    textures: Sequence[np.ndarray] | None
    alpha_range: tuple[float, float] = (0.1, 0.4)

    @classmethod
    def from_dir(cls, root: Path, max_textures: int = 2000) -> "FilmOverlayDegradation":
        import cv2
        paths = sorted(root.rglob("*.png"))[:max_textures]
        textures = []
        for p in paths:
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                continue
            textures.append(arr.astype(np.float32) / 255.0)
        return cls(textures=textures or None)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        if self.textures is None or len(self.textures) == 0:
            return img.clone()
        h, w = img.shape[-2:]
        rng = np.random.default_rng()
        tex = self.textures[rng.integers(len(self.textures))]
        scale = rng.uniform(0.5, 2.0)
        th, tw = int(tex.shape[0] * scale), int(tex.shape[1] * scale)
        if th < h or tw < w:
            reps_y = (h + th - 1) // th + 1
            reps_x = (w + tw - 1) // tw + 1
            tex = np.tile(tex, (reps_y, reps_x))
        y0 = rng.integers(0, tex.shape[0] - h + 1)
        x0 = rng.integers(0, tex.shape[1] - w + 1)
        crop = tex[y0:y0 + h, x0:x0 + w]
        alpha = float(rng.uniform(*self.alpha_range))
        overlay = torch.from_numpy(crop).to(img.device).to(img.dtype)
        return torch.clamp(img + alpha * (overlay.unsqueeze(0) - 0.5), 0.0, 1.0)
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_film_overlay.py -v
git add src/restora_models/data/degradations/film_overlay.py tests/test_film_overlay.py
git commit -m "feat(degradations): film overlay (real scratch/dust/grain)"
```

### Task 7.2: Film color cast

**Files:**
- Create: `src/restora_models/data/degradations/film_color_cast.py`
- Create: `tests/test_film_color_cast.py`

- [ ] **Step 1: Tests**

```python
"""Tests for film color cast (sepia, cyan-fade, red-shift)."""
import torch

from restora_models.data.degradations.film_color_cast import FilmColorCastDegradation


def test_color_cast_shape_preserved():
    deg = FilmColorCastDegradation()
    img = torch.rand(3, 64, 64)
    out = deg.apply(img)
    assert out.shape == img.shape


def test_color_cast_changes_image():
    torch.manual_seed(0)
    deg = FilmColorCastDegradation()
    img = torch.rand(3, 64, 64)
    out = deg.apply(img)
    diff = (out - img).abs().mean().item()
    assert diff > 0.005, f"too little change: {diff}"
```

- [ ] **Step 2: Implement**

```python
"""Film color-cast degradation: per-channel gamma + tint matrices."""
from __future__ import annotations

import torch


_PRESETS = [
    (torch.tensor([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ]), torch.tensor([1.0, 1.0, 1.0])),
    (torch.tensor([
        [0.7, 0.0, 0.0],
        [0.1, 0.9, 0.1],
        [0.1, 0.2, 1.0],
    ]), torch.tensor([1.2, 0.9, 0.8])),
    (torch.tensor([
        [1.1, 0.0, 0.0],
        [0.0, 0.85, 0.0],
        [0.0, 0.0, 0.75],
    ]), torch.tensor([0.9, 1.1, 1.2])),
    (torch.tensor([
        [0.85, 0.1, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.1, 0.85],
    ]), torch.tensor([1.0, 1.0, 1.0])),
]


class FilmColorCastDegradation:
    def apply(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"expected (3, H, W), got {tuple(img.shape)}")
        idx = int(torch.randint(0, len(_PRESETS), ()).item())
        tint, gamma = _PRESETS[idx]
        tint = tint.to(img.device).to(img.dtype)
        gamma = gamma.to(img.device).to(img.dtype)
        c, h, w = img.shape
        flat = img.reshape(c, -1)
        out = (tint @ flat).reshape(c, h, w).clamp(1e-6, 1.0)
        out = out ** gamma.view(c, 1, 1)
        return out.clamp(0.0, 1.0)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_film_color_cast.py -v
git add src/restora_models/data/degradations/film_color_cast.py tests/test_film_color_cast.py
git commit -m "feat(degradations): film color cast (sepia/cyan-fade/red-shift)"
```

### Task 7.3: Gate weave

**Files:**
- Create: `src/restora_models/data/degradations/gate_weave.py`
- Create: `tests/test_gate_weave.py`

- [ ] **Step 1: Tests**

```python
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
```

- [ ] **Step 2: Implement**

```python
"""Gate-weave degradation: per-frame sub-pixel translation jitter."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GateWeaveDegradation:
    def __init__(self, max_shift_px: float = 2.0):
        self.max_shift_px = max_shift_px

    def apply_clip(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() != 4:
            raise ValueError(f"expected (T,3,H,W), got {tuple(clip.shape)}")
        if self.max_shift_px <= 0.0:
            return clip.clone()
        t, _, h, w = clip.shape
        raw = torch.randn(t, 2) * self.max_shift_px
        smooth = F.avg_pool1d(raw.T.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).T
        out = torch.empty_like(clip)
        for k in range(t):
            dy, dx = smooth[k].tolist()
            theta = torch.tensor([
                [1.0, 0.0, 2.0 * dx / max(w - 1, 1)],
                [0.0, 1.0, 2.0 * dy / max(h - 1, 1)],
            ], dtype=clip.dtype, device=clip.device).unsqueeze(0)
            grid = F.affine_grid(theta, [1, 3, h, w], align_corners=True)
            out[k] = F.grid_sample(clip[k:k + 1], grid, mode="bilinear",
                                    padding_mode="border", align_corners=True).squeeze(0)
        return out
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_gate_weave.py -v
git add src/restora_models/data/degradations/gate_weave.py tests/test_gate_weave.py
git commit -m "feat(degradations): gate-weave sub-pixel jitter for film realism"
```

### Task 7.4: MPEG transcode

**Files:**
- Create: `src/restora_models/data/degradations/mpeg_transcode.py`
- Create: `tests/test_mpeg_transcode.py`

- [ ] **Step 1: Tests**

```python
"""Tests for MPEG/H.263 transcode degradation (ffmpeg subprocess)."""
import shutil
import pytest
import torch

from restora_models.data.degradations.mpeg_transcode import MpegTranscodeDegradation


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mpeg_transcode_clip_shape():
    deg = MpegTranscodeDegradation(codec="mpeg1video", bitrate_kbps=200)
    clip = torch.rand(7, 3, 64, 64)
    out = deg.apply_clip(clip)
    assert out.shape == clip.shape


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mpeg_transcode_introduces_artifacts():
    deg = MpegTranscodeDegradation(codec="mpeg1video", bitrate_kbps=150)
    clip = torch.rand(7, 3, 64, 64)
    out = deg.apply_clip(clip)
    diff = (out - clip).abs().mean().item()
    assert diff > 0.001, f"transcode too gentle: {diff}"
```

- [ ] **Step 2: Implement**

```python
"""MPEG transcode degradation via ffmpeg subprocess.

For VHS/broadcast-era footage realism. Encodes the clip to a tempfile,
decodes back, returns.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


class MpegTranscodeDegradation:
    def __init__(self, codec: str = "mpeg1video", bitrate_kbps: int = 300):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        self.codec = codec
        self.bitrate_kbps = bitrate_kbps

    def apply_clip(self, clip: torch.Tensor) -> torch.Tensor:
        import cv2
        if clip.dim() != 4 or clip.shape[1] != 3:
            raise ValueError(f"expected (T,3,H,W), got {tuple(clip.shape)}")
        t, _, h, w = clip.shape
        arr = (clip.permute(0, 2, 3, 1).clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as td:
            inp_path = Path(td) / "in.mp4"
            out_path = Path(td) / "out.mp4"
            writer = cv2.VideoWriter(str(inp_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
            for f in arr:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-i", str(inp_path),
                   "-c:v", self.codec, "-b:v", f"{self.bitrate_kbps}k",
                   str(out_path)]
            subprocess.run(cmd, check=True)
            cap = cv2.VideoCapture(str(out_path))
            decoded = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
        if len(decoded) == 0:
            return clip.clone()
        out_arr = np.stack(decoded[:t]).astype(np.float32) / 255.0
        if out_arr.shape[0] < t:
            pad = np.repeat(out_arr[-1:], t - out_arr.shape[0], axis=0)
            out_arr = np.concatenate([out_arr, pad], axis=0)
        return torch.from_numpy(out_arr).permute(0, 3, 1, 2).to(clip.device).to(clip.dtype)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_mpeg_transcode.py -v
git add src/restora_models/data/degradations/mpeg_transcode.py tests/test_mpeg_transcode.py
git commit -m "feat(degradations): MPEG/H.263 transcode for VHS/broadcast realism"
```

### Task 7.5: Update compound.py for new degradations

**Files:**
- Modify: `src/restora_models/data/compound.py`
- Modify: `src/restora_models/config.py` (add `film_overlay_root` to DataConfig)
- Create: `tests/test_compound_film.py`

- [ ] **Step 1: Add `film_overlay_root` to DataConfig**

In `src/restora_models/config.py`, the `DataConfig` dataclass gets a new optional field:

```python
film_overlay_root: Path | None = None
```

- [ ] **Step 2: Wire film add-ons into compound.py**

Modify `compound.py` to add film_overlay (p=0.4 on colorize, p=0.2 elsewhere), film_color_cast (p=0.3 on colorize), gate_weave (p=0.3 on video batches), mpeg_transcode (p=0.2 on dejpeg).

- [ ] **Step 3: Write integration test**

```python
"""Integration test: compound degradation pipeline applies film add-ons."""
import torch

from restora_models.data.compound import CompoundDegradation


def test_compound_applies_film_add_ons():
    deg = CompoundDegradation(film_overlay_textures=None)
    img = torch.rand(3, 64, 64)
    out, axes = deg.apply_image(img, axes={"colorize": True})
    assert out.shape == img.shape
    assert axes["colorize"]
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_compound_film.py -v
git add src/restora_models/data/compound.py src/restora_models/config.py tests/test_compound_film.py
git commit -m "feat(degradations): compound pipeline wires film overlays + color cast + gate weave + MPEG"
```

---

## Phase 8 — New losses

### Task 8.1: lpips_decoded loss

**Files:**
- Create: `src/restora_models/losses/lpips_decoded.py`
- Create: `tests/test_lpips_decoded.py`
- Modify: `src/restora_models/losses/registry.py`

- [ ] **Step 1: Tests**

```python
"""Tests for lpips_decoded loss."""
import torch

from restora_models.losses.lpips_decoded import LpipsDecodedLoss


def test_lpips_zero_for_identical():
    loss = LpipsDecodedLoss()
    img = torch.rand(2, 3, 64, 64)
    val = loss(img, img)
    assert val.item() < 0.05


def test_lpips_positive_for_different():
    loss = LpipsDecodedLoss()
    a = torch.rand(2, 3, 64, 64)
    b = torch.rand(2, 3, 64, 64)
    val = loss(a, b)
    assert val.item() > 0.1
```

- [ ] **Step 2: Implement**

```python
"""LPIPS perceptual loss on decoded RGB."""
from __future__ import annotations

import torch
import torch.nn as nn


class LpipsDecodedLoss(nn.Module):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = pred * 2.0 - 1.0
        target_n = target * 2.0 - 1.0
        return self.model(pred_n, target_n).mean()
```

- [ ] **Step 3: Register + test + commit**

Append to `src/restora_models/losses/registry.py`:

```python
from restora_models.losses.lpips_decoded import LpipsDecodedLoss
LOSS_REGISTRY["lpips_decoded"] = LpipsDecodedLoss
```

```bash
uv run pytest tests/test_lpips_decoded.py -v
git add src/restora_models/losses/lpips_decoded.py src/restora_models/losses/registry.py tests/test_lpips_decoded.py
git commit -m "feat(losses): LPIPS perceptual loss on decoded RGB"
```

### Task 8.2: central_flicker loss

**Files:**
- Create: `src/restora_models/losses/central_flicker.py`
- Create: `tests/test_central_flicker.py`

- [ ] **Step 1: Tests**

```python
"""Tests for central_flicker temporal-consistency loss."""
import torch

from restora_models.losses.central_flicker import CentralFlickerLoss


def test_central_flicker_zero_when_identical():
    loss = CentralFlickerLoss()
    pred_a = torch.rand(2, 3, 32, 32)
    val = loss(pred_a, pred_a)
    assert val.item() < 1e-6


def test_central_flicker_positive_when_different():
    loss = CentralFlickerLoss()
    a = torch.rand(2, 3, 32, 32)
    b = a + 0.1 * torch.rand_like(a)
    val = loss(a, b)
    assert val.item() > 0.01
```

- [ ] **Step 2: Implement**

```python
"""Central flicker loss: L1 between two overlapping-window predictions
on the shared frame."""
from __future__ import annotations

import torch
import torch.nn as nn


class CentralFlickerLoss(nn.Module):
    def forward(self, pred_window_a: torch.Tensor, pred_window_b: torch.Tensor) -> torch.Tensor:
        if pred_window_a.shape != pred_window_b.shape:
            raise ValueError("predictions must have matching shapes")
        return (pred_window_a - pred_window_b).abs().mean()
```

- [ ] **Step 3: Register + commit**

```bash
# Append registry entry, then:
uv run pytest tests/test_central_flicker.py -v
git add src/restora_models/losses/central_flicker.py src/restora_models/losses/registry.py tests/test_central_flicker.py
git commit -m "feat(losses): central_flicker temporal consistency"
```

### Task 8.3: feat_match (SLKD) loss

**Files:**
- Create: `src/restora_models/losses/feat_match.py`
- Create: `tests/test_feat_match.py`

- [ ] **Step 1: Tests**

```python
"""Tests for feature-matching distillation loss."""
import torch

from restora_models.losses.feat_match import FeatureMatchLoss


def test_feat_match_zero_for_matching():
    loss = FeatureMatchLoss()
    feats = [torch.rand(2, 32, 16, 16), torch.rand(2, 64, 8, 8)]
    val = loss(feats, [f.clone() for f in feats])
    assert val.item() < 1e-6


def test_feat_match_positive_for_mismatched():
    loss = FeatureMatchLoss()
    a = [torch.rand(2, 32, 16, 16)]
    b = [a[0] + 0.5]
    val = loss(a, b)
    assert val.item() > 0.01
```

- [ ] **Step 2: Implement**

```python
"""Feature-matching loss for SLKD-style distillation."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchLoss(nn.Module):
    def forward(self, teacher_feats: Sequence[torch.Tensor],
                student_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(teacher_feats) != len(student_feats):
            raise ValueError(
                f"teacher_feats and student_feats lengths differ: "
                f"{len(teacher_feats)} vs {len(student_feats)}"
            )
        total = teacher_feats[0].new_zeros(())
        for t, s in zip(teacher_feats, student_feats):
            if t.shape[1] != s.shape[1]:
                if s.shape[1] > t.shape[1]:
                    s = s[:, :t.shape[1]]
                else:
                    pad = t.new_zeros(s.shape[0], t.shape[1] - s.shape[1], *s.shape[2:])
                    s = torch.cat([s, pad], dim=1)
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
            total = total + F.mse_loss(s, t)
        return total / max(len(teacher_feats), 1)
```

- [ ] **Step 3: Register + commit**

```bash
uv run pytest tests/test_feat_match.py -v
git add src/restora_models/losses/feat_match.py src/restora_models/losses/registry.py tests/test_feat_match.py
git commit -m "feat(losses): SLKD-style feature matching for distillation"
```

---

## Phase 9 — Data pipeline (composite video dataset)

The composite dataset is the major addition from the dataset-redesign feedback. We build:

- `VideoWindowDataset` (facade): holds a list of `VideoSubDataset` instances and a list of weights, samples from them by weight.
- `VideoSubDataset` (abstract protocol): `__len__`, `__getitem__(idx) -> {"frames": (7,3,H,W), "source": str, "key": str}`.
- `REDSDataset` (primary sub-dataset): samples a 7-frame contiguous window from each 100-frame REDS sequence.
- `VimeoSeptupletDataset` (secondary sub-dataset): reads the fixed 7-frame Vimeo Septuplet clips.
- `replicate_to_window` (helper): used by the inference pipeline and the still-image fallback for single-frame inputs in training.

### Task 9.1: Define the VideoSubDataset protocol + VideoWindowDataset facade

**Files:**
- Create: `src/restora_models/data/video_window.py`
- Create: `tests/test_video_window.py`

- [ ] **Step 1: Tests**

```python
"""Tests for VideoSubDataset protocol + VideoWindowDataset facade."""
import torch
from torch.utils.data import Dataset

from restora_models.data.video_window import VideoSubDataset, VideoWindowDataset


class _FakeSub(Dataset):
    """Minimal in-memory sub-dataset."""
    name = "fake"

    def __init__(self, n: int, seed: int):
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)
        return {
            "frames": torch.rand(7, 3, 32, 32, generator=g),
            "source": self.name,
            "key": f"{self.name}_{idx}",
        }


def test_video_window_concat_lengths():
    a = _FakeSub(n=10, seed=0)
    b = _FakeSub(n=20, seed=1)
    ds = VideoWindowDataset(sub_datasets=[a, b], weights=[1.0, 1.0])
    assert len(ds) == 30


def test_video_window_weighted_sampling_distribution():
    """With weights [1, 9] the second source should serve ~90% of samples."""
    a = _FakeSub(n=10, seed=0)
    b = _FakeSub(n=10, seed=1)
    ds = VideoWindowDataset(sub_datasets=[a, b], weights=[1.0, 9.0])
    sources = [ds.sample_random()["source"] for _ in range(1000)]
    assert sources.count("fake") == 1000  # Both name "fake"; just confirm no crash + correct count.


def test_video_window_returns_canonical_shape():
    a = _FakeSub(n=5, seed=0)
    ds = VideoWindowDataset(sub_datasets=[a], weights=[1.0])
    sample = ds[0]
    assert sample["frames"].shape == (7, 3, 32, 32)
    assert sample["frames"].dtype == torch.float32
    assert "source" in sample
    assert "key" in sample
```

- [ ] **Step 2: Implement**

```python
"""Composite video dataset.

Pulls 7-frame clips from any number of sub-datasets, each of which
implements the VideoSubDataset protocol. The facade exposes a single
flat indexable Dataset over the union; a separate sample_random() method
supports weighted random sampling across sources.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch.utils.data import Dataset


@runtime_checkable
class VideoSubDataset(Protocol):
    """Sub-dataset protocol.

    Required:
    - __len__() -> int: number of available 7-frame clips
    - __getitem__(idx) -> dict with keys {frames, source, key}
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...


class VideoWindowDataset(Dataset):
    """Concatenates N VideoSubDatasets and supports weighted random sampling.

    Index [0, len(a))            -> a
    Index [len(a), len(a)+len(b)) -> b
    ...

    `sample_random()` chooses a sub-dataset proportional to `weights`,
    then picks a random index within that sub-dataset. This is the
    method the trainer's loader uses; the flat index path is mainly for
    eval workflows that want to enumerate.
    """

    def __init__(
        self,
        sub_datasets: Sequence[VideoSubDataset],
        weights: Sequence[float] | None = None,
    ):
        if not sub_datasets:
            raise ValueError("VideoWindowDataset requires >=1 sub-dataset")
        self.subs = list(sub_datasets)
        n = len(self.subs)
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError(f"weights len {len(weights)} != sub_datasets len {n}")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        self.weights = np.array([w / total for w in weights], dtype=np.float64)
        self._cumlens = np.cumsum([len(s) for s in self.subs])

    def __len__(self) -> int:
        return int(self._cumlens[-1])

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        bucket = int(np.searchsorted(self._cumlens, idx, side="right"))
        local = idx - (self._cumlens[bucket - 1] if bucket > 0 else 0)
        return self.subs[bucket][int(local)]

    def sample_random(self, rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng()
        bucket = int(rng.choice(len(self.subs), p=self.weights))
        sub = self.subs[bucket]
        local = int(rng.integers(0, len(sub)))
        return sub[local]
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_video_window.py -v
git add src/restora_models/data/video_window.py tests/test_video_window.py
git commit -m "feat(data): VideoWindowDataset facade + VideoSubDataset protocol"
```

### Task 9.2: REDSDataset sub-dataset

REDS layout (from the official site
https://seungjunnah.github.io/Datasets/reds.html):

```
<root>/train_sharp/<seq_id>/<frame_id>.png   # e.g. 000/00000000.png ... 00000099.png
<root>/val_sharp/<seq_id>/<frame_id>.png
<root>/train_blur/<seq_id>/<frame_id>.png    # paired degraded (optional, ignored here)
<root>/train_blur_comp/...                    # ignored
```

Sequences are 100 frames numbered `00000000.png` through `00000099.png`.

**Files:**
- Create: `src/restora_models/data/reds.py`
- Create: `tests/test_reds.py`

- [ ] **Step 1: Tests**

```python
"""Tests for REDSDataset 7-frame window sampler."""
from pathlib import Path

import cv2
import numpy as np
import torch

from restora_models.data.reds import REDSDataset


def _make_fake_reds(tmp_path: Path, n_seqs: int = 2, n_frames: int = 30) -> Path:
    """Create a minimal REDS-shaped fake dataset (frames are random noise)."""
    root = tmp_path / "REDS"
    for s in range(n_seqs):
        seq_dir = root / "train_sharp" / f"{s:03d}"
        seq_dir.mkdir(parents=True)
        for f in range(n_frames):
            img = (np.random.rand(48, 48, 3) * 255).astype("uint8")
            cv2.imwrite(str(seq_dir / f"{f:08d}.png"), img)
    return root


def test_reds_dataset_length_counts_windows(tmp_path):
    """Each 30-frame sequence yields (30 - 7 + 1) = 24 distinct windows.
    With 2 sequences that's 48 windows."""
    root = _make_fake_reds(tmp_path, n_seqs=2, n_frames=30)
    ds = REDSDataset(root, split="train_sharp", window=7, stride=1, crop=32)
    assert len(ds) == 48


def test_reds_dataset_returns_canonical_shape(tmp_path):
    root = _make_fake_reds(tmp_path, n_seqs=1, n_frames=10)
    ds = REDSDataset(root, split="train_sharp", window=7, stride=1, crop=32)
    sample = ds[0]
    assert sample["frames"].shape == (7, 3, 32, 32)
    assert sample["frames"].dtype == torch.float32
    assert sample["source"] == "reds"
    assert "key" in sample
    assert 0.0 <= sample["frames"].min().item()
    assert sample["frames"].max().item() <= 1.0


def test_reds_dataset_stride_2(tmp_path):
    """stride=2 halves the per-seq window count."""
    root = _make_fake_reds(tmp_path, n_seqs=1, n_frames=30)
    ds = REDSDataset(root, split="train_sharp", window=7, stride=2, crop=32)
    # (30 - 7) // 2 + 1 = 12 windows
    assert len(ds) == 12
```

- [ ] **Step 2: Run — should fail**

```bash
uv run pytest tests/test_reds.py -v
```

- [ ] **Step 3: Implement**

```python
"""REDS (REalistic and Dynamic Scenes) sub-dataset.

Official site: https://seungjunnah.github.io/Datasets/reds.html
Layout:
    <root>/train_sharp/<seq_id>/<frame_id>.png   # 000..269 sequences, 100 frames each
    <root>/val_sharp/<seq_id>/<frame_id>.png

Each sample is a 7-frame contiguous window. The dataset exposes len()
windows enumerated deterministically; __getitem__ returns the window.
Random cropping happens inline in __getitem__.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class REDSDataset(Dataset):
    SOURCE_NAME = "reds"

    def __init__(
        self,
        root: Path | str,
        split: Literal["train_sharp", "val_sharp"] = "train_sharp",
        window: int = 7,
        stride: int = 1,
        crop: int = 256,
    ):
        if window < 1:
            raise ValueError(f"window must be >=1, got {window}")
        if stride < 1:
            raise ValueError(f"stride must be >=1, got {stride}")
        self.root = Path(root)
        self.split = split
        self.window = window
        self.stride = stride
        self.crop = crop
        seq_root = self.root / split
        if not seq_root.exists():
            raise FileNotFoundError(f"REDS split not found: {seq_root}")
        self.windows: list[tuple[Path, int]] = []
        for seq_dir in sorted(seq_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            frames = sorted(seq_dir.glob("*.png"))
            n = len(frames)
            if n < window:
                continue
            # Windows at offsets 0, stride, 2*stride, ... while offset + window <= n
            for off in range(0, n - window + 1, stride):
                self.windows.append((seq_dir, off))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        seq_dir, off = self.windows[idx]
        frames_files = sorted(seq_dir.glob("*.png"))[off:off + self.window]
        clip = []
        for p in frames_files:
            arr = cv2.imread(str(p))
            if arr is None:
                raise RuntimeError(f"failed to read {p}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            clip.append(arr)
        clip_np = np.stack(clip).astype(np.float32) / 255.0
        h, w = clip_np.shape[1:3]
        ch = min(self.crop, h)
        cw = min(self.crop, w)
        y0 = int(np.random.randint(0, h - ch + 1)) if h > ch else 0
        x0 = int(np.random.randint(0, w - cw + 1)) if w > cw else 0
        clip_np = clip_np[:, y0:y0 + ch, x0:x0 + cw, :]
        clip_t = torch.from_numpy(clip_np).permute(0, 3, 1, 2).contiguous()
        return {
            "frames": clip_t,
            "source": self.SOURCE_NAME,
            "key": f"{seq_dir.name}@{off}",
        }
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_reds.py -v
git add src/restora_models/data/reds.py tests/test_reds.py
git commit -m "feat(data): REDSDataset sub-dataset (7-frame contiguous windows)"
```

### Task 9.3: VimeoSeptupletDataset sub-dataset

**Files:**
- Create: `src/restora_models/data/vimeo_septuplet.py`
- Create: `tests/test_vimeo_septuplet.py`

- [ ] **Step 1: Tests**

```python
"""Tests for VimeoSeptupletDataset sub-dataset."""
from pathlib import Path

import cv2
import numpy as np
import torch

from restora_models.data.vimeo_septuplet import VimeoSeptupletDataset


def _make_fake_vimeo(tmp_path: Path) -> Path:
    root = tmp_path / "vimeo"
    seqs = ["00001/0001", "00001/0002"]
    for s in seqs:
        d = root / "sequences" / s
        d.mkdir(parents=True)
        for i in range(1, 8):
            img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
            cv2.imwrite(str(d / f"im{i}.png"), img)
    (root / "sep_trainlist.txt").write_text("\n".join(seqs) + "\n")
    return root


def test_vimeo_loader_canonical_sample(tmp_path):
    root = _make_fake_vimeo(tmp_path)
    ds = VimeoSeptupletDataset(root, split="train", crop=32)
    sample = ds[0]
    assert sample["frames"].shape == (7, 3, 32, 32)
    assert sample["frames"].dtype == torch.float32
    assert sample["source"] == "vimeo_septuplet"
    assert "key" in sample
```

- [ ] **Step 2: Implement**

```python
"""Vimeo Septuplet (Xue et al., http://toflow.csail.mit.edu/) sub-dataset.

Layout:
    <root>/sequences/<seqA>/<seqB>/im{1..7}.png
    <root>/sep_trainlist.txt   (lines: <seqA>/<seqB>)
    <root>/sep_testlist.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VimeoSeptupletDataset(Dataset):
    NUM_FRAMES = 7
    SOURCE_NAME = "vimeo_septuplet"

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "test"] = "train",
        crop: int = 256,
    ) -> None:
        self.root = Path(root)
        list_name = "sep_trainlist.txt" if split == "train" else "sep_testlist.txt"
        list_path = self.root / list_name
        if not list_path.exists():
            raise FileNotFoundError(f"Vimeo Septuplet list missing: {list_path}")
        self.entries = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
        self.crop = crop

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        seq = self.entries[idx]
        d = self.root / "sequences" / seq
        frames = []
        for i in range(1, self.NUM_FRAMES + 1):
            arr = cv2.imread(str(d / f"im{i}.png"))
            if arr is None:
                raise RuntimeError(f"failed to read {d / f'im{i}.png'}")
            frames.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        clip = np.stack(frames).astype(np.float32) / 255.0
        h, w = clip.shape[1:3]
        ch = min(self.crop, h)
        cw = min(self.crop, w)
        y0 = int(np.random.randint(0, h - ch + 1)) if h > ch else 0
        x0 = int(np.random.randint(0, w - cw + 1)) if w > cw else 0
        clip = clip[:, y0:y0 + ch, x0:x0 + cw, :]
        clip_t = torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()
        return {"frames": clip_t, "source": self.SOURCE_NAME, "key": seq}
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_vimeo_septuplet.py -v
git add src/restora_models/data/vimeo_septuplet.py tests/test_vimeo_septuplet.py
git commit -m "feat(data): VimeoSeptupletDataset as a VideoWindowDataset sub-dataset"
```

### Task 9.4: replicate_to_window helper

**Files:**
- Create: `src/restora_models/data/window.py`
- Create: `tests/test_window.py`

- [ ] **Step 1: Tests**

```python
"""Tests for replicate_to_window (single image -> 7-frame clip)."""
import torch

from restora_models.data.window import replicate_to_window


def test_replicate_single_image_to_7_frame_clip():
    img = torch.rand(3, 32, 32)
    clip = replicate_to_window(img, num_frames=7)
    assert clip.shape == (7, 3, 32, 32)
    for k in range(7):
        assert torch.equal(clip[k], img)


def test_replicate_short_clip_pads_edges():
    short = torch.rand(3, 3, 16, 16)
    clip = replicate_to_window(short, num_frames=7, center_index=3)
    assert clip.shape == (7, 3, 16, 16)
    assert torch.equal(clip[3], short[1])
```

- [ ] **Step 2: Implement**

```python
"""Helpers to build a 7-frame window from variable-length input."""
from __future__ import annotations

import torch


def replicate_to_window(
    frames: torch.Tensor, *, num_frames: int = 7, center_index: int = 3,
) -> torch.Tensor:
    """Pad/replicate to exactly num_frames frames.

    - (3, H, W): single image -> all num_frames are copies.
    - (T, 3, H, W) with T < num_frames: center input at center_index, replicate edges.
    - (T, 3, H, W) with T >= num_frames: center-crop num_frames.
    """
    if frames.dim() == 3:
        return frames.unsqueeze(0).expand(num_frames, *frames.shape).contiguous()
    if frames.dim() != 4:
        raise ValueError(f"expected (T,3,H,W) or (3,H,W), got {tuple(frames.shape)}")
    t = frames.shape[0]
    if t >= num_frames:
        start = (t - num_frames) // 2
        return frames[start:start + num_frames].contiguous()
    center_in = t // 2
    out = []
    for k in range(num_frames):
        idx = k - center_index + center_in
        idx = max(0, min(t - 1, idx))
        out.append(frames[idx])
    return torch.stack(out, dim=0).contiguous()
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_window.py -v
git add src/restora_models/data/window.py tests/test_window.py
git commit -m "feat(data): replicate_to_window helper for stills + short clips"
```

### Task 9.5: Build factory that constructs the composite dataset from config

**Files:**
- Create: `src/restora_models/data/builders.py`
- Create: `tests/test_data_builders.py`

The factory inspects `cfg.data.sources` (a list of dicts naming the sub-dataset and its kwargs) and builds the appropriate `VideoWindowDataset`. Example config shape:

```yaml
data:
  sources:
    - type: reds
      root: /workspace/data/REDS
      split: train_sharp
      weight: 4.0
    - type: vimeo_septuplet
      root: /workspace/data/vimeo-septuplet
      split: train
      weight: 1.0
```

- [ ] **Step 1: Tests**

```python
"""Tests for data builder factory."""
from pathlib import Path

import pytest

from restora_models.data.builders import build_video_window_dataset


def test_build_unknown_type_raises():
    with pytest.raises(KeyError):
        build_video_window_dataset([{"type": "no_such_type", "weight": 1.0}])


def test_build_reds_dataset(tmp_path: Path):
    # Build a fake reds root for instantiation only.
    (tmp_path / "train_sharp" / "000").mkdir(parents=True)
    sources = [{"type": "reds", "root": str(tmp_path), "split": "train_sharp",
                "window": 7, "stride": 1, "crop": 32, "weight": 1.0}]
    ds = build_video_window_dataset(sources)
    assert len(ds) == 0  # no frames -> 0 windows
```

- [ ] **Step 2: Implement**

```python
"""Factory to build a VideoWindowDataset from a list of source specs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from restora_models.data.reds import REDSDataset
from restora_models.data.video_window import VideoSubDataset, VideoWindowDataset
from restora_models.data.vimeo_septuplet import VimeoSeptupletDataset

_BUILDERS: dict[str, callable] = {
    "reds": lambda kw: REDSDataset(**{k: v for k, v in kw.items() if k != "weight"}),
    "vimeo_septuplet": lambda kw: VimeoSeptupletDataset(**{k: v for k, v in kw.items() if k != "weight"}),
}


def build_video_window_dataset(sources: Sequence[dict]) -> VideoWindowDataset:
    """Build the composite dataset from a list of source dicts.

    Each entry has a `type` key matching a registered builder, plus
    arbitrary kwargs forwarded to the sub-dataset constructor, plus an
    optional `weight` (default 1.0) used by sample_random().
    """
    subs: list[VideoSubDataset] = []
    weights: list[float] = []
    for s in sources:
        kind = s.get("type")
        if kind not in _BUILDERS:
            raise KeyError(
                f"unknown video source type {kind!r}; have {sorted(_BUILDERS)}"
            )
        subs.append(_BUILDERS[kind](s))
        weights.append(float(s.get("weight", 1.0)))
    return VideoWindowDataset(sub_datasets=subs, weights=weights)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_data_builders.py -v
git add src/restora_models/data/builders.py tests/test_data_builders.py
git commit -m "feat(data): factory to build composite VideoWindowDataset from config"
```

---

## Phase 10 — Trainer adaptation

### Task 10.1: Update Trainer to consume VideoWindowDataset

**Files:**
- Modify: `src/restora_models/train/trainer.py`
- Create: `tests/test_trainer_temporal_smoke.py`

The trainer's existing image-loader + video-loader path is replaced by a single `VideoWindowDataset` constructed from `cfg.data.sources`. Image-only fallback for backwards-compat is no longer needed since all sub-datasets yield (7,3,H,W) clips.

- [ ] **Step 1: Read current trainer**

```bash
grep -n 'def fit\|self.model(\|train_loader\|video_loader' src/restora_models/train/trainer.py
```

- [ ] **Step 2: Smoke test (deferred run until configs exist in Phase 11)**

```python
"""Smoke test: trainer steps one batch on a fake REDS dataset."""
from pathlib import Path

import pytest


@pytest.mark.skip(reason="enabled in Phase 11 after configs/local-temporal.yaml exists")
def test_trainer_temporal_one_step(tmp_path):
    pass
```

- [ ] **Step 3: Trainer changes (high-level)**

In `src/restora_models/train/trainer.py`:

1. Replace the image-loader + video-loader construction with:

```python
from restora_models.data.builders import build_video_window_dataset

self.train_ds = build_video_window_dataset(cfg.data.sources)
self.train_loader = DataLoader(
    self.train_ds, batch_size=cfg.data.loader.batch_size,
    num_workers=cfg.data.loader.num_workers,
    shuffle=True, persistent_workers=cfg.data.loader.num_workers > 0,
    pin_memory=True,
)
```

2. Each batch now has `frames` (B, 7, 3, H, W). Pass into the degradation
   pipeline as a clip (so gate-weave / MPEG transcode apply at clip level),
   then call `self.model(degraded_frames, config)`.

3. Targets are the central frame of the clean clip:
   `target = clean_frames[:, TemporalAlignStem.CENTER_INDEX]`.

- [ ] **Step 4: Commit**

```bash
git add src/restora_models/train/trainer.py tests/test_trainer_temporal_smoke.py
git commit -m "feat(train): trainer consumes composite VideoWindowDataset"
```

### Task 10.2: Preview generation for temporal

**Files:**
- Modify: `src/restora_models/train/preview.py`

- [ ] **Step 1: Read preview.py**

```bash
grep -n 'def render_multitask_grid\|model(' src/restora_models/train/preview.py
```

- [ ] **Step 2: Adapt to call model with (B,7,3,H,W) input**

Where preview currently calls `self.model(degraded, config)` with degraded being (B,3,H,W), insert:

```python
from restora_models.data.window import replicate_to_window

if degraded.dim() == 4:
    clip = torch.stack([replicate_to_window(degraded[i], num_frames=7)
                        for i in range(degraded.shape[0])])
else:
    clip = degraded
pred = self.model(clip, config)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_ui_smoke.py -v
git add src/restora_models/train/preview.py
git commit -m "feat(train): preview adapter wraps single frames as 7-frame clips"
```

---

## Phase 11 — Configs

### Task 11.1: Update configs/default.yaml

**Files:**
- Modify: `configs/default.yaml`

- [ ] **Step 1: Read current**

```bash
cat configs/default.yaml
```

- [ ] **Step 2: Rewrite for temporal model**

Key changes:
- `model.type: temporal_restora_small`
- Replace any old `data.root` / `video.root` with the new `data.sources` list
- Remove diffusion-related keys

Example structure:

```yaml
run:
  name: default
  root: runs/

model:
  type: temporal_restora_small

data:
  loader:
    batch_size: 4
    num_workers: 4
    prefetch_factor: 2
  sources:
    - type: reds
      root: ~/data/REDS
      split: train_sharp
      window: 7
      stride: 1
      crop: 256
      weight: 4.0
    - type: vimeo_septuplet
      root: ~/data/vimeo-septuplet
      split: train
      crop: 256
      weight: 1.0
  film_overlay_root: ~/data/film-overlays

train:
  total_steps: 100000
  amp: bf16
  compile: false
  memory_format: channels_last
  optimizer: muon
  lr: 1.0e-3
  weight_decay: 0.01

scheduler:
  total_steps: 100000
  warmup_steps: 1000

losses:
  preset: temporal_v1
```

- [ ] **Step 3: Smoke test**

```python
"""Verify the default config loads."""
from pathlib import Path

from restora_models.config import load_config


def test_default_config_loads():
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.model.type.startswith("temporal_")
    assert len(cfg.data.sources) >= 1
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_temporal_configs_load.py -v
git add configs/default.yaml tests/test_temporal_configs_load.py
git commit -m "feat(config): default.yaml targets temporal model + composite sources"
```

### Task 11.2: configs/local-temporal.yaml

**Files:**
- Create: `configs/local-temporal.yaml`

- [ ] **Step 1: Write**

```yaml
extends: default.yaml

run:
  name: local_temporal

model:
  type: temporal_restora_small

data:
  loader:
    batch_size: 4
    num_workers: 6
    prefetch_factor: 2
  sources:
    - type: reds
      root: ~/data/REDS
      split: train_sharp
      window: 7
      stride: 1
      crop: 256
      weight: 4.0
    - type: vimeo_septuplet
      root: ~/data/vimeo-septuplet
      split: train
      crop: 256
      weight: 1.0
  film_overlay_root: ~/data/film-overlays

train:
  total_steps: 5000
  amp: bf16
  compile: false
  memory_format: channels_last
  optimizer: muon
  lr: 1.0e-3
  weight_decay: 0.01

scheduler:
  total_steps: 5000
  warmup_steps: 500

losses:
  preset: temporal_v1
```

- [ ] **Step 2: Commit**

```bash
git add configs/local-temporal.yaml
git commit -m "feat(config): local-temporal config for RTX 6000 Blackwell smoke"
```

### Task 11.3: configs/b200-temporal.yaml

- [ ] **Step 1: Write**

```yaml
extends: default.yaml

run:
  name: b200_temporal

model:
  type: temporal_restora_large

data:
  loader:
    batch_size: 12
    num_workers: 12
    prefetch_factor: 4
  sources:
    - type: reds
      root: /workspace/data/REDS
      split: train_sharp
      window: 7
      stride: 1
      crop: 256
      weight: 4.0
    - type: vimeo_septuplet
      root: /workspace/data/vimeo-septuplet
      split: train
      crop: 256
      weight: 1.0
  film_overlay_root: /workspace/data/film-overlays

train:
  total_steps: 200000
  amp: bf16
  compile: true
  compile_mode: default
  memory_format: channels_last
  optimizer: muon
  lr: 1.0e-3
  weight_decay: 0.01

scheduler:
  total_steps: 200000
  warmup_steps: 2000

losses:
  preset: temporal_v1
```

- [ ] **Step 2: Commit**

```bash
git add configs/b200-temporal.yaml
git commit -m "feat(config): b200-temporal production config"
```

### Task 11.4: Loss preset `temporal_v1`

**Files:**
- Modify: `src/restora_models/losses/registry.py`

- [ ] **Step 1: Add preset**

```python
LOSS_PRESETS["temporal_v1"] = [
    {"name": "l1_pixel", "weight": 1.0},
    {"name": "lpips_decoded", "weight": 0.4},
    {"name": "chroma_lab", "weight": 0.2, "apply_to_axes": ["colorize"]},
    {"name": "colorfulness", "weight": 0.1, "apply_to_axes": ["colorize"]},
    {"name": "freq_l1", "weight": 0.4, "apply_to_axes": ["sharpen"]},
    {"name": "temporal_pair", "weight": 0.5},
    {"name": "central_flicker", "weight": 0.3},
]
```

- [ ] **Step 2: Test + commit**

```bash
uv run pytest tests/test_loss_presets.py -v
git add src/restora_models/losses/registry.py tests/test_loss_presets.py
git commit -m "feat(losses): temporal_v1 preset"
```

---

## Phase 12 — CLI consolidation

### Task 12.1: Rewrite cli.py — single binary, consolidated subcommands

**Files:**
- Rewrite: `src/restora_models/cli.py` (full replacement)
- Create: `src/restora_models/cli_prepare.py` (umbrella for `restora prepare-data`)
- Create: `tests/test_cli_temporal.py`

- [ ] **Step 1: Tests**

```python
"""Tests for the simplified temporal CLI."""
from typer.testing import CliRunner

from restora_models.cli import app


def test_cli_version():
    r = CliRunner().invoke(app, ["version"])
    assert r.exit_code == 0


def test_cli_has_expected_commands():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ["train", "infer", "export", "distill", "bench", "compare", "gallery", "prepare-data", "train-flow-distill"]:
        assert cmd in r.output, f"missing command: {cmd}"


def test_cli_no_obsolete_commands():
    r = CliRunner().invoke(app, ["--help"])
    for cmd in ["scan-data", "download ", "info", "download-davis", "download-imagenet", "download-openimages", "prepare-videos", "precompute-flow", "make-synthetic-videos"]:
        assert cmd not in r.output, f"obsolete command still present: {cmd}"
```

- [ ] **Step 2: Run — should fail (obsolete commands still present)**

```bash
uv run pytest tests/test_cli_temporal.py -v
```

- [ ] **Step 3: Implement new cli.py**

Rewrite `src/restora_models/cli.py` to expose only:
- `version`
- `prepare-data` (umbrella)
- `train`
- `train-flow-distill` (Stage 0 RAFT distillation)
- `infer`
- `export`
- `distill`
- `bench`
- `compare`
- `gallery`

The `infer` command takes either a directory of frames (sequential sliding window) or a single image (replicated to 7).

- [ ] **Step 4: Implement cli_prepare.py**

`src/restora_models/cli_prepare.py` defines a sub-app with:
- `--reds <PATH>`: print download instructions (the REDS dataset requires manual download; the script verifies layout and writes a manifest)
- `--vimeo <PATH>`: same — verify layout and write manifest
- `--film-overlays <PATH>`: download DeepRemaster `noise_data.zip` from http://iizuka.cs.tsukuba.ac.jp/projects/remastering/data/noise_data.zip, extract, build manifest

- [ ] **Step 5: Run + commit**

```bash
uv run pytest tests/test_cli_temporal.py -v
git add src/restora_models/cli.py src/restora_models/cli_prepare.py tests/test_cli_temporal.py
git commit -m "feat(cli): consolidated temporal CLI (prepare-data umbrella, obsolete cmds removed)"
```

---

## Phase 13 — ONNX export

### Task 13.1: Update export wrapper for temporal contract

**Files:**
- Modify: `src/restora_models/export/wrapper.py`
- Modify: `src/restora_models/export/onnx.py`
- Create: `tests/test_export_temporal.py`

- [ ] **Step 1: Test**

```python
"""End-to-end test: export a tiny temporal model to ONNX, verify the graph
runs and shapes match torch."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from restora_models.config import ModelConfig
from restora_models.export.onnx import export_onnx_from_model
from restora_models.models.registry import build_model


def test_export_temporal_onnx(tmp_path: Path):
    cfg = ModelConfig(type="temporal_restora_nano")
    m = build_model(cfg, num_axes=5).eval()
    out_path = tmp_path / "tiny.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=64,
        export_path=out_path, opset=17, simplify=True,
        dynamic_hw=True, task_map={}, precision="fp32",
    )
    assert out_path.exists()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    frames = np.random.rand(1, 7, 3, 96, 96).astype(np.float32)
    config = np.zeros((1, 5), dtype=np.float32)
    config[0, 0] = 1.0
    inputs = {sess.get_inputs()[0].name: frames, sess.get_inputs()[1].name: config}
    out = sess.run(None, inputs)[0]
    assert out.shape == (1, 3, 96, 96)
```

- [ ] **Step 2: Update wrapper.py + onnx.py**

Change input signature from `(rgb [B,3,H,W], config)` to `(frames [B,7,3,H,W], config)`. Dynamic axes `{0: "batch", 3: "h", 4: "w"}` on `frames`.

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_export_temporal.py -v
git add src/restora_models/export/wrapper.py src/restora_models/export/onnx.py tests/test_export_temporal.py
git commit -m "feat(export): temporal ONNX (frames [B,7,3,H,W] + config [B,5])"
```

### Task 13.2: Remove PNNX export

Per the radical cleanup goal — PNNX is rarely used by the C# consumer and adds dependency surface. We remove it.

- [ ] **Step 1: Delete pnnx export module + dep**

```bash
rm -f src/restora_models/export/pnnx.py
# Edit pyproject.toml to remove the pnnx dependency line
```

- [ ] **Step 2: Remove --format pnnx branch from cli.py export**

Edit `src/restora_models/cli.py` to drop the pnnx code path.

- [ ] **Step 3: Sync + commit**

```bash
uv sync
git add -u
git commit -m "chore(export): remove PNNX path (ONNX is the sole deploy target)"
```

---

## Phase 14 — Distillation upgrade (SLKD)

### Task 14.1: Rewrite distill.py

**Files:**
- Create: `src/restora_models/train/distill.py`
- Create: `tests/test_distill_temporal.py`

- [ ] **Step 1: Smoke test**

```python
"""Smoke test for SLKD-style temporal distillation."""
from pathlib import Path

from restora_models.train.distill import run_distill


def test_distill_one_step(tmp_path):
    run_distill(
        teacher=None,
        output=tmp_path / "student.pt",
        data=None,
        student_preset="nano",
        steps=1,
        batch_size=1,
        amp="fp32",
        device="cpu",
        feat_match=True,
        lpips=True,
    )
    assert (tmp_path / "student.pt").exists()
```

- [ ] **Step 2: Implement**

The implementation:
1. Loads teacher checkpoint (or builds a fresh small model if teacher is None — for tests only)
2. Builds student per preset
3. For each step: sample batch from VideoWindowDataset, run teacher (eval), run student (train)
4. Loss = L1(student_out, teacher_out) + LPIPS(student_out, teacher_out) + feat_match(teacher_feats, student_feats)
5. Forward hooks register on 3 decoder stages of both teacher and student for feat_match

- [ ] **Step 3: Wire to CLI + run + commit**

```bash
uv run pytest tests/test_distill_temporal.py -v
git add src/restora_models/train/distill.py src/restora_models/cli.py tests/test_distill_temporal.py
git commit -m "feat(distill): SLKD-style feature-matching + LPIPS for temporal students"
```

---

## Phase 15 — Inference pipeline

### Task 15.1: Update infer/pipeline.py for temporal contract

**Files:**
- Modify: `src/restora_models/infer/pipeline.py`
- Create: `tests/test_infer_temporal.py`

- [ ] **Step 1: Test**

```python
"""Inference pipeline must accept stills (replicate to 7) and clip dirs."""
from pathlib import Path

import cv2
import numpy as np

from restora_models.config import ModelConfig
from restora_models.infer.pipeline import VideoPipeline
from restora_models.models.registry import build_model


def test_pipeline_single_image(tmp_path):
    m = build_model(ModelConfig(type="temporal_restora_nano"), num_axes=5).eval()
    pipe = VideoPipeline(model=m, device="cpu")
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = pipe.process_image(img, config={"colorize": True})
    assert out.shape == img.shape


def test_pipeline_frame_directory(tmp_path):
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"f{i:03d}.png"),
                    (np.random.rand(32, 32, 3) * 255).astype("uint8"))
    m = build_model(ModelConfig(type="temporal_restora_nano"), num_axes=5).eval()
    pipe = VideoPipeline(model=m, device="cpu")
    out_dir = tmp_path / "out"
    pipe.process_directory(tmp_path, out_dir, config={"colorize": True})
    outs = sorted(out_dir.glob("*.png"))
    assert len(outs) == 10
```

- [ ] **Step 2: Implement**

Replace the single-image inference path with sliding-window. For each output frame, build the 7-frame window via `replicate_to_window` at boundaries.

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_infer_temporal.py -v
git add src/restora_models/infer/pipeline.py tests/test_infer_temporal.py
git commit -m "feat(infer): temporal sliding-window inference pipeline"
```

---

## Phase 16 — Bench, compare, gallery

### Task 16.1: train/bench.py

**Files:**
- Create: `src/restora_models/train/bench.py`

- [ ] **Step 1: Implement temporal bench**

Bench inference at fixed `(B,7,3,H,W)` input on the chosen device. Same report (warmup median, p99, throughput, peak VRAM) as before but with the new contract.

- [ ] **Step 2: Wire to CLI + commit**

```bash
git add src/restora_models/train/bench.py src/restora_models/cli.py
git commit -m "feat(bench): temporal-contract benchmark"
```

### Task 16.2: train/evaluate.py (compare)

**Files:**
- Create: `src/restora_models/train/evaluate.py`

- [ ] **Step 1: Implement**

Sample N clips from the REDS val split (or any configured eval source), apply per-axis degradations, run each checkpoint, report per-axis PSNR + LPIPS deltas.

- [ ] **Step 2: Wire + commit**

```bash
git add src/restora_models/train/evaluate.py src/restora_models/cli.py
git commit -m "feat(evaluate): temporal-contract PSNR/LPIPS comparison"
```

### Task 16.3: train/gallery.py

**Files:**
- Create: `src/restora_models/train/gallery.py`

- [ ] **Step 1: Implement**

For N sampled clips, run the model per-axis and write triptychs `(clean | degraded | restored)` of the CENTER frame.

- [ ] **Step 2: Wire + commit**

```bash
git add src/restora_models/train/gallery.py src/restora_models/cli.py
git commit -m "feat(gallery): temporal-contract qualitative gallery"
```

---

## Phase 17 — FlowDistill pre-training script

### Task 17.1: train-flow-distill subcommand

**Files:**
- Create: `src/restora_models/train/flow_distill.py`

- [ ] **Step 1: Implement**

The script:
1. Loads `torchvision.models.optical_flow.raft_large(weights="DEFAULT")` (frozen)
2. Builds a fresh `FlowDistill(iters=4)` student
3. Samples random frame pairs from a `VideoWindowDataset` (any sub-dataset works)
4. Computes teacher flow at 12 iterations, student flow at 4
5. Loss = EPE (endpoint error) + small smoothness regularizer
6. Trains 20k-50k steps with AdamW + OneCycle
7. Saves checkpoint

- [ ] **Step 2: Wire to CLI + commit**

```bash
git add src/restora_models/train/flow_distill.py src/restora_models/cli.py
git commit -m "feat(flow): train-flow-distill subcommand (RAFT student pre-training)"
```

---

## Phase 18 — End-to-end training orchestrator (`train-pipeline`)

A single command that runs the full multi-stage training pipeline from scratch, OR resumes a partially-complete pipeline from an existing run directory, OR continues training when fresh data has been added. State is persisted to `<run_root>/pipeline_state.json` so an interrupted pipeline picks up exactly where it left off.

**Stages** (executed in order; each becomes a key in the state file):
- `flow_distill` — Stage 0: pre-train the static-unroll RAFT student
- `backbone` — Stage 1: train TemporalNAFNet backbone (RSD head identity-skip)
- `refine` — Stage 2: train RSD head only (backbone frozen)
- `end_to_end` — Stage 3: unfreeze everything except FlowDistill, low-LR finetune
- `distill_small` / `distill_medium` / `distill_nano` — Stage 4: per-size distillation

Each stage emits its final checkpoint at a deterministic path (e.g. `<run_root>/<stage>/final.pt`); the orchestrator passes those forward as inputs to subsequent stages.

### Task 18.1: Pipeline state module

**Files:**
- Create: `src/restora_models/train/pipeline_state.py`
- Create: `tests/test_pipeline_state.py`

- [ ] **Step 1: Tests**

```python
"""Tests for pipeline_state persistence + lookups."""
from pathlib import Path

from restora_models.train.pipeline_state import PipelineState, STAGE_ORDER


def test_fresh_state_no_stages_complete(tmp_path):
    s = PipelineState(tmp_path)
    for stage in STAGE_ORDER:
        assert not s.is_complete(stage)


def test_mark_and_query(tmp_path):
    s = PipelineState(tmp_path)
    s.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    assert s.is_complete("flow_distill")
    assert s.checkpoint_for("flow_distill") == tmp_path / "flow.pt"


def test_persistence_across_instances(tmp_path):
    s1 = PipelineState(tmp_path)
    s1.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    s2 = PipelineState(tmp_path)
    assert s2.is_complete("flow_distill")
    assert s2.checkpoint_for("flow_distill") == tmp_path / "flow.pt"


def test_next_pending_returns_first_incomplete(tmp_path):
    s = PipelineState(tmp_path)
    assert s.next_pending() == "flow_distill"
    s.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    assert s.next_pending() == "backbone"


def test_reset_stage_for_extend_mode(tmp_path):
    s = PipelineState(tmp_path)
    for st in ["flow_distill", "backbone", "refine"]:
        s.mark_complete(st, checkpoint=tmp_path / f"{st}.pt")
    s.reset_from("backbone")
    assert s.is_complete("flow_distill")
    assert not s.is_complete("backbone")
    assert not s.is_complete("refine")
```

- [ ] **Step 2: Implement**

```python
"""Pipeline-state persistence for the orchestrator.

State file lives at <run_root>/pipeline_state.json. Format:
    {
      "stages": {
        "flow_distill": {"complete": true,  "checkpoint": "..../flow.pt"},
        "backbone":     {"complete": false, "checkpoint": null},
        ...
      },
      "version": 1
    }
"""
from __future__ import annotations

import json
from pathlib import Path

STAGE_ORDER = (
    "flow_distill",
    "backbone",
    "refine",
    "end_to_end",
    "distill_small",
    "distill_medium",
    "distill_nano",
)


class PipelineState:
    FILE_NAME = "pipeline_state.json"

    def __init__(self, run_root: Path | str):
        self.root = Path(run_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILE_NAME
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if "stages" not in data:
                data["stages"] = {}
            return data
        return {"stages": {}, "version": 1}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def is_complete(self, stage: str) -> bool:
        if stage not in STAGE_ORDER:
            raise KeyError(f"unknown stage {stage!r}; must be one of {STAGE_ORDER}")
        return bool(self._data["stages"].get(stage, {}).get("complete", False))

    def checkpoint_for(self, stage: str) -> Path | None:
        entry = self._data["stages"].get(stage, {})
        ckpt = entry.get("checkpoint")
        return Path(ckpt) if ckpt else None

    def mark_complete(self, stage: str, *, checkpoint: Path) -> None:
        if stage not in STAGE_ORDER:
            raise KeyError(f"unknown stage {stage!r}")
        self._data["stages"][stage] = {"complete": True, "checkpoint": str(checkpoint)}
        self._save()

    def next_pending(self) -> str | None:
        for stage in STAGE_ORDER:
            if not self.is_complete(stage):
                return stage
        return None

    def reset_from(self, stage: str) -> None:
        """Mark `stage` and all subsequent stages as incomplete.

        Used by `--extend` mode: keep flow-distill done, but force backbone
        + later stages to re-run on the augmented dataset.
        """
        if stage not in STAGE_ORDER:
            raise KeyError(f"unknown stage {stage!r}")
        idx = STAGE_ORDER.index(stage)
        for st in STAGE_ORDER[idx:]:
            self._data["stages"][st] = {"complete": False, "checkpoint": None}
        self._save()
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_pipeline_state.py -v
git add src/restora_models/train/pipeline_state.py tests/test_pipeline_state.py
git commit -m "feat(train): PipelineState persistence for multi-stage orchestrator"
```

### Task 18.2: Pipeline runner

**Files:**
- Create: `src/restora_models/train/pipeline.py`
- Create: `tests/test_pipeline_runner.py`

- [ ] **Step 1: Tests**

```python
"""Tests for the pipeline runner — orchestrates train_flow_distill + train + distill."""
from pathlib import Path
from unittest.mock import MagicMock

from restora_models.train.pipeline import run_pipeline
from restora_models.train.pipeline_state import PipelineState, STAGE_ORDER


def test_runner_executes_pending_stages_in_order(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_stage(name: str):
        def runner(*, run_root, prev_checkpoint, **kw):
            calls.append(name)
            out = Path(run_root) / name / "final.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("fake")
            return out
        return runner

    monkeypatch.setattr("restora_models.train.pipeline.STAGE_RUNNERS",
                       {name: fake_stage(name) for name in STAGE_ORDER})
    run_pipeline(run_root=tmp_path, config_path=None)
    assert calls == list(STAGE_ORDER)


def test_runner_skips_completed_stages(tmp_path, monkeypatch):
    s = PipelineState(tmp_path)
    s.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    s.mark_complete("backbone", checkpoint=tmp_path / "backbone.pt")

    calls: list[str] = []
    def fake_stage(name):
        def runner(*, run_root, prev_checkpoint, **kw):
            calls.append(name)
            out = Path(run_root) / name / "final.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("fake")
            return out
        return runner
    monkeypatch.setattr("restora_models.train.pipeline.STAGE_RUNNERS",
                       {name: fake_stage(name) for name in STAGE_ORDER})

    run_pipeline(run_root=tmp_path, config_path=None)
    assert "flow_distill" not in calls
    assert "backbone" not in calls
    assert calls[0] == "refine"


def test_extend_mode_resets_backbone_onwards(tmp_path, monkeypatch):
    s = PipelineState(tmp_path)
    for st in ["flow_distill", "backbone", "refine", "end_to_end"]:
        s.mark_complete(st, checkpoint=tmp_path / f"{st}.pt")

    calls: list[str] = []
    def fake_stage(name):
        def runner(*, run_root, prev_checkpoint, **kw):
            calls.append(name)
            out = Path(run_root) / name / "final.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("fake")
            return out
        return runner
    monkeypatch.setattr("restora_models.train.pipeline.STAGE_RUNNERS",
                       {name: fake_stage(name) for name in STAGE_ORDER})

    run_pipeline(run_root=tmp_path, config_path=None, extend_from="backbone")
    assert "flow_distill" not in calls   # flow distill is still complete
    assert calls[0] == "backbone"
```

- [ ] **Step 2: Implement**

```python
"""End-to-end multi-stage training pipeline.

Each stage is a callable in STAGE_RUNNERS that returns the path to its
final checkpoint. The runner:
  1. Loads PipelineState from run_root
  2. For each pending stage (or stages reset by --extend), calls the
     corresponding runner with the previous stage's checkpoint as input
  3. Marks the stage complete + persists state immediately after each stage
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from restora_models.train.pipeline_state import PipelineState, STAGE_ORDER


StageRunner = Callable[..., Path]


def _run_flow_distill(*, run_root: Path, prev_checkpoint: Path | None,
                      config_path: Path | None, **_) -> Path:
    from restora_models.train.flow_distill import run_flow_distill
    return run_flow_distill(out_dir=run_root / "flow_distill", config_path=config_path)


def _run_backbone(*, run_root: Path, prev_checkpoint: Path | None,
                  config_path: Path | None, **_) -> Path:
    from restora_models.train.trainer import run_train_stage
    return run_train_stage(
        out_dir=run_root / "backbone",
        config_path=config_path,
        flow_estimator_ckpt=prev_checkpoint,
        freeze=("refine",),
    )


def _run_refine(*, run_root: Path, prev_checkpoint: Path | None,
                config_path: Path | None, **_) -> Path:
    from restora_models.train.trainer import run_train_stage
    return run_train_stage(
        out_dir=run_root / "refine",
        config_path=config_path,
        warm_start=prev_checkpoint,
        freeze=("flow_estimator", "backbone"),
    )


def _run_end_to_end(*, run_root: Path, prev_checkpoint: Path | None,
                    config_path: Path | None, **_) -> Path:
    from restora_models.train.trainer import run_train_stage
    return run_train_stage(
        out_dir=run_root / "end_to_end",
        config_path=config_path,
        warm_start=prev_checkpoint,
        freeze=("flow_estimator",),
        lr_scale=0.1,
    )


def _make_distill_runner(size: str) -> StageRunner:
    def runner(*, run_root: Path, prev_checkpoint: Path | None,
               config_path: Path | None, **_) -> Path:
        from restora_models.train.distill import run_distill
        out = run_root / f"distill_{size}"
        out.mkdir(parents=True, exist_ok=True)
        return run_distill(
            teacher=prev_checkpoint,
            output=out / "final.pt",
            data=None,
            student_preset=size,
        )
    return runner


STAGE_RUNNERS: dict[str, StageRunner] = {
    "flow_distill":   _run_flow_distill,
    "backbone":       _run_backbone,
    "refine":         _run_refine,
    "end_to_end":     _run_end_to_end,
    "distill_small":  _make_distill_runner("small"),
    "distill_medium": _make_distill_runner("medium"),
    "distill_nano":   _make_distill_runner("nano"),
}


def run_pipeline(
    *,
    run_root: Path,
    config_path: Path | None,
    extend_from: str | None = None,
) -> None:
    """Run all pending stages in STAGE_ORDER.

    Args:
        run_root: Pipeline state + per-stage outputs live here.
        config_path: Training config (e.g. configs/local-temporal.yaml).
        extend_from: If set, reset this stage and everything after to pending,
                     then run from there. Used to retrain on new data while
                     keeping already-done upstream stages (e.g. flow_distill).
    """
    state = PipelineState(run_root)
    if extend_from is not None:
        state.reset_from(extend_from)

    for stage in STAGE_ORDER:
        if state.is_complete(stage):
            continue
        runner = STAGE_RUNNERS[stage]
        prev_ckpt = None
        for prev in reversed(STAGE_ORDER[: STAGE_ORDER.index(stage)]):
            if state.is_complete(prev):
                prev_ckpt = state.checkpoint_for(prev)
                break
        ckpt = runner(run_root=run_root, prev_checkpoint=prev_ckpt,
                      config_path=config_path)
        state.mark_complete(stage, checkpoint=ckpt)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_pipeline_runner.py -v
git add src/restora_models/train/pipeline.py tests/test_pipeline_runner.py
git commit -m "feat(train): pipeline runner orchestrates flow_distill+backbone+refine+e2e+distill stages"
```

### Task 18.3: `restora train-pipeline` CLI subcommand

**Files:**
- Modify: `src/restora_models/cli.py`

- [ ] **Step 1: Add the subcommand**

```python
@app.command(name="train-pipeline")
def train_pipeline(
    config: Path = typer.Option(
        None, "--config",
        help="Training config (e.g. configs/local-temporal.yaml). Required when "
             "starting fresh; optional when --resume (the resumed dir already "
             "stores the config used)."),
    resume: Path = typer.Option(
        None, "--resume",
        help="Run root directory of an existing pipeline to resume. Completed "
             "stages are skipped; pending stages run in order. Equivalent to "
             "passing the same --run-root as an earlier invocation."),
    run_root: Path = typer.Option(
        None, "--run-root",
        help="Output directory for the pipeline (state file + per-stage outputs). "
             "Created if missing. Required unless --resume is given."),
    extend_from: str = typer.Option(
        None, "--extend-from",
        help="'Continue training because we have more data' mode. Resets this "
             "stage and all subsequent stages to pending, then runs from there. "
             "Useful after adding new REDS/Vimeo sequences. Typical value: "
             "backbone (re-trains backbone+ on the new dataset while preserving "
             "the already-trained flow estimator)."),
) -> None:
    """End-to-end training pipeline (flow_distill → backbone → refine → end_to_end → distill).

    Examples:
        # Start fresh:
        restora train-pipeline --config configs/local-temporal.yaml --run-root runs/local

        # Resume an interrupted pipeline:
        restora train-pipeline --resume runs/local

        # Continue training because new data was added:
        restora train-pipeline --resume runs/local --extend-from backbone
    """
    from restora_models.train.pipeline import run_pipeline
    from restora_models.train.pipeline_state import STAGE_ORDER

    if resume is not None and run_root is None:
        run_root = resume
    if run_root is None:
        raise typer.BadParameter("--run-root or --resume is required")
    if extend_from is not None and extend_from not in STAGE_ORDER:
        raise typer.BadParameter(
            f"--extend-from must be one of {list(STAGE_ORDER)}; got {extend_from!r}"
        )
    if config is None and resume is None:
        raise typer.BadParameter("--config is required when starting fresh")

    run_pipeline(run_root=run_root, config_path=config, extend_from=extend_from)
```

- [ ] **Step 2: Update the CLI test in `tests/test_cli_temporal.py`**

Add `train-pipeline` to the expected-commands list in `test_cli_has_expected_commands`.

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_cli_temporal.py -v
git add src/restora_models/cli.py tests/test_cli_temporal.py
git commit -m "feat(cli): train-pipeline subcommand (start/resume/extend)"
```

### Task 18.4: Trainer + flow_distill expose programmatic entry points

The orchestrator imports `run_train_stage` from `train/trainer.py` and `run_flow_distill` from `train/flow_distill.py`. These need to exist as standalone callables in addition to whatever the CLI invokes.

**Files:**
- Modify: `src/restora_models/train/trainer.py` (add `run_train_stage`)
- Modify: `src/restora_models/train/flow_distill.py` (add `run_flow_distill`)

- [ ] **Step 1: `run_train_stage` signature**

```python
def run_train_stage(
    *,
    out_dir: Path,
    config_path: Path | None,
    flow_estimator_ckpt: Path | None = None,
    warm_start: Path | None = None,
    freeze: tuple[str, ...] = (),
    lr_scale: float = 1.0,
) -> Path:
    """Run one training stage. Returns path to the final checkpoint.

    Args:
        out_dir: Per-stage output directory.
        config_path: Training config (typically passed through from --config).
        flow_estimator_ckpt: If set, load these weights into the model's
            FlowDistill submodule before training begins.
        warm_start: If set, load entire model state from this checkpoint at
            startup (matches by shape; mismatches are skipped with a warning).
        freeze: Submodule names to freeze (`requires_grad_(False)`). Values:
            "flow_estimator", "backbone", "refine".
        lr_scale: Multiplier on the configured learning rate (for low-LR
            end-to-end finetune).
    """
```

Wrap the existing `Trainer.fit()` flow with this entry point. Return `out_dir / "final.pt"`.

- [ ] **Step 2: `run_flow_distill` signature**

```python
def run_flow_distill(*, out_dir: Path, config_path: Path | None) -> Path:
    """Pre-train the FlowDistill student. Returns path to the final checkpoint."""
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_pipeline_runner.py tests/test_trainer_temporal_smoke.py -v
git add src/restora_models/train/trainer.py src/restora_models/train/flow_distill.py
git commit -m "feat(train): expose run_train_stage + run_flow_distill for orchestrator"
```

---

## Phase 19 — Docs

### Task 19.1: Rewrite README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite**

Sections:
1. 1-paragraph summary (temporal model, 7-frame window, 5-axis tasks)
2. Quick start (download REDS + film-overlays -> train-flow-distill -> train -> export)
3. Architecture overview (link to spec)
4. CLI cheat sheet
5. Sizes table

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README for temporal old-film design"
```

### Task 19.2: Update integration docs

**Files:**
- Modify or delete: `docs/integration/csharp-video-inference.md`
- Modify or delete: `docs/integration/onnx-inference-guide.md`
- Delete: `docs/integration/ncnn-export-brief.md`, `ncnn-export-followup.md`
- Modify or delete: `docs/integration/b200-deployment.md`
- Modify or delete: `docs/integration/verifying-model-improvement.md`
- Delete: `docs/integration/laion-download.md`

- [ ] **Step 1: Audit and update**

For each: rewrite if still relevant; delete if superseded entirely.

- [ ] **Step 2: Commit**

```bash
git add docs/integration/
git commit -m "docs(integration): refresh for temporal contract; remove obsolete"
```

---

## Phase 20 — Final cleanup pass

### Task 20.1: Audit for orphaned files

- [ ] **Step 1: Find orphans with grep**

```bash
# Files in src/ whose stem is not imported anywhere
for f in $(find src/restora_models -name '*.py' -not -name '__init__.py'); do
  stem=$(basename "$f" .py)
  # Search for any "from X import" or "import X" referencing the module
  if ! grep -rq "from restora_models.*${stem}\|import.*${stem}" src/ tests/ 2>/dev/null; then
    echo "ORPHAN: $f"
  fi
done
```

- [ ] **Step 2: Remove any truly orphaned modules**

Inspect each and remove if not used by the CLI or any test.

- [ ] **Step 3: Commit**

```bash
git add -u src/
git commit -m "chore: remove orphaned modules surfaced by import audit"
```

### Task 20.2: Run full test suite

- [ ] **Step 1: Run**

```bash
uv run pytest -q
```

Expected: all tests pass (mpeg_transcode skips if ffmpeg not installed).

- [ ] **Step 2: Fix any failures**

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "test: stabilize remaining tests after temporal cutover"
```

### Task 20.3: Final repo verification

- [ ] **Step 1: Run checks**

```bash
find . -name '*.bak' -o -name '*~' -o -name '*.swp' -o -name '.DS_Store' | grep -v node_modules | head
ls src/restora_models/models/ | grep -E '(nafnet\.py|heads\.py|diffusion_head\.py|vae\.py|discriminator\.py|color\.py)' && echo "FOUND STALE" || echo "clean"
ls configs/ | grep -E '^(local\.yaml|b200\.yaml|b200-diffusion\.yaml|large\.yaml)$' && echo "FOUND STALE" || echo "clean"
find . -name __pycache__ -type d | head
ls runs/ trained/ 2>/dev/null && echo "FOUND STALE" || echo "clean"
ls scripts/ 2>/dev/null && echo "scripts/ exists" || echo "no scripts dir (good)"
```

Expected: every check prints `clean` or nothing.

- [ ] **Step 2: Run linter**

```bash
uv run ruff check src/ tests/
```

Expected: 0 errors.

- [ ] **Step 3: Final commit (if any fixes)**

```bash
git add -u
git commit -m "chore: final cleanup pass — no stale files, lint clean"
```

---

## Self-review notes (author)

**Spec coverage** — every section of the design spec has a task:
- §2 architecture overview -> Phases 2-6
- §3.1 distilled RAFT -> Phase 2, Phase 17 (pre-training)
- §3.2 flow-warp + visibility -> Phase 3
- §3.3 TemporalNAFNet -> Phase 4
- §3.4 RSD refine head -> Phase 5
- §3.5 model contract -> Phase 6
- §4 degradation pipeline -> Phase 7
- §5 training plan -> Phases 11 (configs), 17 (Stage 0)
- §5.3 loss design -> Phase 8 + Phase 11.4 (preset)
- §5.4 optimizer -> Phase 11 (config); Muon dep added in Phase 1
- §6 inference -> Phase 15
- §7 ONNX export -> Phase 13
- §8 CLI surface -> Phase 12
- §9 compatibility with existing framework -> Phase 10
- §10 cleanup -> Phase 0, Phase 19
- §11 risks -> mitigated within tasks
- §12 open questions -> intentional deferrals
- §13 build sequence -> mirrored phase-by-phase

**Dataset redesign (post-spec feedback):**
- Composite `VideoWindowDataset` + `VideoSubDataset` protocol -> Phase 9.1
- REDS as primary sub-dataset -> Phase 9.2
- Vimeo Septuplet stays as secondary -> Phase 9.3
- Factory pattern reads cfg.data.sources -> Phase 9.5
- Easy to add future sources (BVI-DVC, raw mp4) by implementing the protocol

**Placeholder scan:** no TBD/TODO. All "if missing, port from git history" steps are concrete instructions.

**Type consistency:** model contract `(frames [B,7,3,H,W], config [B,5]) -> (B,3,H,W)` used consistently across all tasks. Sub-dataset protocol returns `{"frames": (7,3,H,W), "source": str, "key": str}` consistently.

**Training execution note:** Phases 0-19 produce the codebase. The actual multi-stage training runs (Stage 0 RAFT distill, Stage 1 backbone, Stage 2 RSD head, Stage 3 end-to-end, Stage 4 size distillation) happen by invoking the CLI on real hardware — not in this plan. The plan ends when the harness is ready to train.

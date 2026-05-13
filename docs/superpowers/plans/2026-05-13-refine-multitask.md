# Refine Multi-Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified multi-task image restoration codebase (`refine`) that handles colorization, denoising, super-resolution x2/x4, deblur, and JPEG-restore in one network, lifting v1's training infrastructure and replacing the colorization-specific bits.

**Architecture:** NAFNet U-Net (4 encoder downsamples, /16 bottleneck with transformer) + LAB conversion `nn.Module`s baked in + FiLM/AdaLN task conditioning + residual learning in normalized LAB. Single RGB-in, RGB-out forward; 2-input ONNX (rgb, task_id). Model behind a `@register_model` registry so future architectures plug in cleanly.

**Tech Stack:** Python 3.11, `uv`, PyTorch 2.4+, `typer`, `pydantic`, `pyyaml`, `rich`, `opencv-python-headless`, `onnx`/`onnxruntime`/`onnxsim`/`onnxscript`, `pytest`. Notably: no `timm` (no ConvNeXt encoder this time).

**Spec:** `docs/superpowers/specs/2026-05-13-refine-multitask-design.md`

---

## File Structure

```
coliraz/                                    # the existing git repo
├── legacy/coliraz-v1/                      # archived v1
│   └── (everything that was at repo root for v1)
├── reference/ddcolor_original/             # untouched
├── runs/                                   # untouched
├── docs/                                   # stays at root (sub-docs are still useful)
├── src/refine/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── color.py                        # RgbToLab + LabToRgb nn.Modules
│   │   ├── task_embed.py
│   │   ├── nafblock.py
│   │   ├── transformer_block.py
│   │   └── nafnet.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── pixel.py
│   │   ├── perceptual.py
│   │   ├── gan.py
│   │   ├── colorfulness.py
│   │   ├── freq.py
│   │   └── metrics.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── multitask.py
│   │   └── degradations/
│   │       ├── __init__.py
│   │       ├── registry.py
│   │       ├── colorization.py
│   │       ├── denoise.py
│   │       ├── superres.py
│   │       ├── deblur.py
│   │       └── jpeg.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── ui.py
│   │   ├── preview.py
│   │   ├── ema.py
│   │   └── checkpoint.py
│   ├── infer/__init__.py + pipeline.py
│   ├── export/__init__.py + onnx.py
│   └── utils/
│       ├── __init__.py
│       ├── color.py                        # pure rgb<->lab functions
│       ├── gpu.py
│       └── timing.py
├── configs/{default.yaml, tiny.yaml, large.yaml, laion-multitask.yaml}
├── tests/  (16 tests)
├── pyproject.toml
├── main.py
└── README.md
```

## Conventions

- All tests CPU-only; full suite under 30 s. Slow gates use `REFINE_SLOW=1`.
- Each commit prefixed `refine:` and ends with the Co-Authored-By line.
- Every task ends with `uv run pytest -q` confirming no regressions before commit.
- Each task is independently shippable on its own.

---

## Task 1: Archive v1, scaffold v2

**Files:**
- Move: every v1 file at repo root → `legacy/coliraz-v1/`
- Create: `pyproject.toml`, `main.py`, `README.md`, `.gitignore`, `src/refine/__init__.py` (and stub `__init__.py` in every subpackage), `tests/conftest.py`, `tests/__init__.py`

- [ ] **Step 1: Move v1 files**

```bash
mkdir -p legacy/coliraz-v1
git mv src configs tests pyproject.toml main.py uv.lock legacy/coliraz-v1/
ls legacy/coliraz-v1/  # should show src/ configs/ tests/ pyproject.toml main.py uv.lock
```

- [ ] **Step 2: Add archive note to legacy README**

Write `legacy/coliraz-v1/README.md`:

```markdown
# coliraz v1 (archived)

This is the v1 colorization-only project. It is preserved unchanged.
To resurrect:

```sh
cd legacy/coliraz-v1
uv sync --extra dev
uv run coliraz train --config configs/laion-large-vivid.yaml --data <path>
```

For the active multi-task v2, see ../../README.md.
```

- [ ] **Step 3: Write new top-level `pyproject.toml`**

```toml
[project]
name = "refine"
version = "0.1.0"
description = "Multi-task image restoration: colorization, SR, denoise, deblur, JPEG"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.4",
  "torchvision>=0.19",
  "opencv-python-headless>=4.10",
  "numpy>=1.26,<3",
  "pillow>=10",
  "typer>=0.12",
  "pydantic>=2.7",
  "pyyaml>=6.0",
  "rich>=13.7",
  "onnx>=1.16",
  "onnxruntime>=1.19",
  "onnxsim>=0.4",
  "onnxscript>=0.1",
  "nvidia-ml-py>=12.0",
  "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff>=0.6"]

[project.scripts]
refine = "refine.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/refine"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

- [ ] **Step 4: Top-level `main.py`**

```python
"""Thin entry point — delegates to the Typer CLI."""
from refine.cli import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Top-level `README.md`**

```markdown
# refine

Multi-task image restoration: one model trained jointly on colorization,
super-resolution, denoising, deblurring, and JPEG-artifact removal.

## Quick start

```sh
uv sync
uv run refine scan-data --root /path/to/images
uv run refine train --config configs/laion-multitask.yaml --data /path/to/images
```

See `docs/superpowers/specs/2026-05-13-refine-multitask-design.md` for the design.

The previous colorization-only project lives in `legacy/coliraz-v1/`.
```

- [ ] **Step 6: `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
wheels/

# Virtual env / package mgmt
.venv/
.uv/

# Test / lint caches
.pytest_cache/
.ruff_cache/

# Refine outputs
runs/
samples/
.refine-manifest.txt
.coliraz-manifest.txt
*.onnx
.cache/
```

- [ ] **Step 7: Package skeletons**

```bash
for d in src/refine src/refine/models src/refine/losses src/refine/data \
         src/refine/data/degradations src/refine/train src/refine/infer \
         src/refine/export src/refine/utils tests; do
  mkdir -p "$d" && touch "$d/__init__.py"
done
```

`src/refine/__init__.py`:

```python
"""refine — multi-task image restoration."""
__version__ = "0.1.0"
```

- [ ] **Step 8: Stub CLI**

`src/refine/cli.py`:

```python
import typer

app = typer.Typer(help="refine — multi-task image restoration", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """refine CLI."""


@app.command()
def version() -> None:
    from refine import __version__
    typer.echo(__version__)
```

- [ ] **Step 9: `tests/conftest.py`**

```python
import pathlib

import numpy as np
import pytest


@pytest.fixture
def tmp_image_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """tmp dir with 6 small synthetic RGB images in a nested tree."""
    import cv2

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    rng = np.random.default_rng(0)
    paths = [
        tmp_path / "img0.png",
        tmp_path / "img1.jpg",
        tmp_path / "a" / "img2.png",
        tmp_path / "a" / "img3.webp",
        tmp_path / "a" / "b" / "img4.jpeg",
        tmp_path / "a" / "b" / "img5.bmp",
    ]
    for p in paths:
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(p), img)
    return tmp_path


@pytest.fixture
def small_image_uint8() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
```

- [ ] **Step 10: Install and smoke test**

```bash
uv sync --extra dev
uv run refine version  # expect: 0.1.0
uv run pytest -q       # expect: "no tests ran"
```

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refine: archive coliraz v1 + scaffold refine v2

Move all v1 source/configs/tests to legacy/coliraz-v1/ with a brief
README explaining how to resurrect it. Scaffold the new refine
project at repo root with uv-managed pyproject.toml, empty package
tree, Typer CLI stub, and conftest fixtures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Color conversion functions (pure)

**Files:**
- Create: `src/refine/utils/color.py`
- Create: `tests/test_color.py`

Same pure-tensor `rgb_to_lab` / `lab_to_rgb` / `derive_gray_rgb_from_rgb` as v1. These will be wrapped in `nn.Module`s in Task 4.

- [ ] **Step 1: Test `tests/test_color.py`**

```python
import cv2
import numpy as np
import torch

from refine.utils.color import (
    derive_gray_rgb_from_rgb,
    lab_to_rgb,
    rgb_to_lab,
)


def test_rgb_to_lab_matches_cv2(small_image_uint8):
    rgb_f32 = small_image_uint8.astype(np.float32) / 255.0
    expected = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2LAB)
    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)
    got = rgb_to_lab(t).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, expected, atol=1.0)


def test_lab_to_rgb_round_trip(small_image_uint8):
    rgb_f32 = small_image_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)
    back = lab_to_rgb(rgb_to_lab(t)).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(back, rgb_f32, atol=0.02)


def test_derive_gray_rgb_matches_reference(small_image_uint8):
    rgb_f32 = small_image_uint8.astype(np.float32) / 255.0
    L = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2LAB)[:, :, :1]
    expected = cv2.cvtColor(
        np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1),
        cv2.COLOR_LAB2RGB,
    )
    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)
    got = derive_gray_rgb_from_rgb(t).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, expected, atol=0.01)
```

- [ ] **Step 2: Implement `src/refine/utils/color.py`**

(Copy verbatim from `legacy/coliraz-v1/src/coliraz/utils/color.py` — the math is identical.)

```python
"""Vectorized RGB <-> LAB conversion on torch tensors. Matches cv2 float32 convention."""
from __future__ import annotations

import torch

_RGB2XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ2RGB = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])  # D65


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=0).pow(1 / 2.4) - 0.055)


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(t > delta**3, t.clamp(min=0).pow(1.0 / 3.0), t / (3 * delta**2) + 4.0 / 29.0)


def _f_lab_inv(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(t > delta, t.pow(3), 3 * delta**2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(rgb.shape)}")
    m = _RGB2XYZ.to(rgb.device, dtype=rgb.dtype)
    w = _WHITE.to(rgb.device, dtype=rgb.dtype)
    xyz = torch.einsum("ij,bjhw->bihw", m, _srgb_to_linear(rgb)) / w.view(1, 3, 1, 1)
    f = _f_lab(xyz)
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    if lab.dim() != 4 or lab.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(lab.shape)}")
    m = _XYZ2RGB.to(lab.device, dtype=lab.dtype)
    w = _WHITE.to(lab.device, dtype=lab.dtype)
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = torch.cat([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], dim=1) * w.view(1, 3, 1, 1)
    return _linear_to_srgb(torch.einsum("ij,bjhw->bihw", m, xyz))


def derive_gray_rgb_from_rgb(rgb: torch.Tensor) -> torch.Tensor:
    lab = rgb_to_lab(rgb)
    L = lab[:, 0:1]
    return lab_to_rgb(torch.cat([L, torch.zeros_like(L), torch.zeros_like(L)], dim=1)).clamp(0, 1)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_color.py -q   # expect: 3 passed
git add src/refine/utils/color.py tests/test_color.py
git commit -m "refine: tensor RGB<->LAB conversion (lifted from v1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: GPU + timing utilities

**Files:**
- Create: `src/refine/utils/gpu.py`
- Create: `src/refine/utils/timing.py`
- Create: `tests/test_utils_misc.py`

Lifted from v1 verbatim.

- [ ] **Step 1: Test `tests/test_utils_misc.py`**

```python
import time

from refine.utils.gpu import GpuStats, gpu_stats
from refine.utils.timing import EMA, Stopwatch


def test_gpu_stats_returns_none_or_dataclass():
    s = gpu_stats(0)
    assert s is None or isinstance(s, GpuStats)


def test_ema_smooths_values():
    e = EMA(alpha=0.5)
    assert e.update(10.0) == 10.0
    assert e.update(20.0) == 15.0
    assert e.update(20.0) == 17.5


def test_stopwatch_measures_time():
    sw = Stopwatch().start()
    time.sleep(0.01)
    assert sw.stop() > 0.005
```

- [ ] **Step 2: `src/refine/utils/gpu.py`**

(Identical to v1's `coliraz/utils/gpu.py`.)

```python
"""Optional pynvml-backed GPU stats. Returns None on any error / missing dep."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GpuStats:
    name: str
    mem_used_gb: float
    mem_total_gb: float
    util_pct: int
    temp_c: int
    power_w: float
    power_limit_w: float


_HANDLE_CACHE: dict[int, object] = {}
_INITIALIZED = False


def _ensure_init() -> bool:
    global _INITIALIZED
    if _INITIALIZED:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
        _INITIALIZED = True
        return True
    except Exception:
        return False


def gpu_stats(device_index: int = 0) -> GpuStats | None:
    if not _ensure_init():
        return None
    try:
        import pynvml
        h = _HANDLE_CACHE.get(device_index)
        if h is None:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            _HANDLE_CACHE[device_index] = h
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        try:
            plim = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
        except Exception:
            plim = 0.0
        return GpuStats(name=name, mem_used_gb=mem.used / 1024**3, mem_total_gb=mem.total / 1024**3,
                        util_pct=int(util), temp_c=int(temp), power_w=pw, power_limit_w=plim)
    except Exception:
        return None
```

- [ ] **Step 3: `src/refine/utils/timing.py`**

```python
"""Tiny EMA + Stopwatch."""
from __future__ import annotations

import time


class EMA:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1 - self.alpha) * self.value
        return self.value


class Stopwatch:
    def __init__(self) -> None:
        self._t0: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        assert self._t0 is not None
        self.elapsed = time.perf_counter() - self._t0
        return self.elapsed
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_utils_misc.py -q   # 3 passed
git add src/refine/utils/gpu.py src/refine/utils/timing.py tests/test_utils_misc.py
git commit -m "refine: gpu/timing utilities (lifted from v1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Color conversion `nn.Module`s

**Files:**
- Create: `src/refine/models/color.py`
- Create: `tests/test_color_modules.py`

These wrap the pure functions with `autocast(enabled=False)` for bf16 stability (the lesson we hard-won in v1). They also do the normalization to ~N(0,1) so downstream blocks can run in bf16 cleanly.

- [ ] **Step 1: Test**

```python
import torch

from refine.models.color import LabToRgb, RgbToLab


def test_rgb_to_lab_module_shape():
    m = RgbToLab()
    rgb = torch.rand(2, 3, 16, 16)
    lab_n = m(rgb)
    assert lab_n.shape == rgb.shape


def test_round_trip_through_modules():
    rgb = torch.rand(2, 3, 16, 16)
    lab_n = RgbToLab()(rgb)
    rgb_back = LabToRgb()(lab_n)
    assert rgb_back.shape == rgb.shape
    # very loose tolerance because the normalization is lossy at extremes
    assert (rgb_back - rgb).abs().mean() < 0.05


def test_fp32_dispatch_under_bf16_autocast():
    """Even when autocast(bf16) is active, conversion runs in fp32 and
    returns fp32 output (no NaN/Inf)."""
    m = RgbToLab()
    rgb = torch.rand(1, 3, 8, 8)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = m(rgb)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Implement `src/refine/models/color.py`**

```python
"""nn.Module wrappers around the pure color conversions.

These are frozen (no learned parameters) and always run in fp32. The
bf16 autocast is explicitly disabled inside the forward — pow(x, 1/2.4)
and the lab inverse functions overflow bf16 dynamic range, which was
the dominant training-instability failure mode in coliraz v1.

The modules also include a fixed normalization so the LAB output has
roughly N(0, 1) statistics, which lets downstream NAFBlocks process
under bf16 cleanly.
"""
from __future__ import annotations

import torch
from torch import nn

from refine.utils.color import lab_to_rgb, rgb_to_lab


class RgbToLab(nn.Module):
    """(B, 3, H, W) RGB in [0, 1]  →  (B, 3, H, W) normalized LAB fp32."""

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(rgb.device.type, enabled=False):
            lab = rgb_to_lab(rgb.float())
            L = (lab[:, 0:1] - 50.0) / 50.0  # → ~[-1, 1]
            a = lab[:, 1:2] / 110.0  # → ~[-1.2, 1.2]
            b = lab[:, 2:3] / 110.0
            return torch.cat([L, a, b], dim=1)


class LabToRgb(nn.Module):
    """(B, 3, H, W) normalized LAB fp32  →  (B, 3, H, W) RGB clamped [0, 1]."""

    def forward(self, lab_n: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(lab_n.device.type, enabled=False):
            x = lab_n.float()
            L = x[:, 0:1] * 50.0 + 50.0
            a = x[:, 1:2] * 110.0
            b = x[:, 2:3] * 110.0
            lab = torch.cat([L, a, b], dim=1)
            return lab_to_rgb(lab).clamp(0, 1)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_color_modules.py -q   # 3 passed
git add src/refine/models/color.py tests/test_color_modules.py
git commit -m "refine: RgbToLab/LabToRgb nn.Modules with frozen fp32 dispatch

Wraps the pure conversions with autocast(enabled=False) so they never
run in bf16 (the v1 NaN bug). Adds normalization to ~N(0,1) so the
network operates in a numerically benign range internally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Pydantic config + YAML loader

**Files:**
- Create: `src/refine/config.py`
- Create: `tests/test_config.py`

Extends v1's config with `DegradationConfig`, `TaskMaskedLossConfig` (adds `apply_to_tasks`), and `model.type` so the model registry can pick a backbone.

- [ ] **Step 1: Test `tests/test_config.py`**

```python
from pathlib import Path

import pytest

from refine.config import Config, deep_merge, expand_loss_preset, load_config


def test_preset_minimal():
    losses = expand_loss_preset("minimal")
    assert [l.name for l in losses] == ["l1_rgb"]


def test_preset_standard_has_colorfulness_for_colorize_only():
    losses = expand_loss_preset("standard")
    cf = [l for l in losses if l.name == "colorfulness"]
    assert len(cf) == 1
    assert cf[0].apply_to_tasks == ["colorize"]


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    over = {"a": {"b": 99}}
    assert deep_merge(base, over) == {"a": {"b": 99, "c": 2}, "d": 3}


def test_load_config_with_preset(tmp_path: Path):
    (tmp_path / "x.yaml").write_text(
        "data: { root: /tmp/x }\nlosses: !preset minimal\n"
        "degradations: { colorize: { weight: 1.0 } }\n"
    )
    cfg = load_config(tmp_path / "x.yaml")
    assert isinstance(cfg, Config)
    assert cfg.data.root == "/tmp/x"
    assert [l.name for l in cfg.losses] == ["l1_rgb"]


def test_chained_defaults(tmp_path: Path):
    (tmp_path / "base.yaml").write_text(
        "data: { root: /a }\nlosses: !preset minimal\n"
        "degradations: { colorize: { weight: 1.0 } }\n"
    )
    (tmp_path / "child.yaml").write_text("defaults: base.yaml\ndata: { val_fraction: 0.05 }\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.data.root == "/a"
    assert cfg.data.val_fraction == 0.05


def test_required_fields_raise():
    with pytest.raises(Exception):
        Config.model_validate({})
```

- [ ] **Step 2: Implement `src/refine/config.py`**

```python
"""Pydantic v2 config + YAML loader with chained defaults and !preset tag."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# ---------- model ----------------------------------------------------------

class ModelConfig(BaseModel):
    type: str = "nafnet"
    size: Literal["tiny", "large"] = "tiny"
    input_size: int = 256  # used only as tracing example for ONNX export
    nf: int | None = None  # overrides size preset if set
    enc_depths: list[int] | None = None
    bottle_blocks: int | None = None
    hidden_dim: int | None = None
    task_embed_dim: int = 128


# ---------- data -----------------------------------------------------------

class AugmentConfig(BaseModel):
    hflip: bool = True
    rotate90: bool = False


class LoaderConfig(BaseModel):
    batch_size: int | Literal["auto"] = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


class DataConfig(BaseModel):
    root: str
    val_fraction: float = 0.01
    num_fixed_preview_samples: int = 2
    num_random_preview_samples: int = 1
    augment: AugmentConfig = AugmentConfig()
    loader: LoaderConfig = LoaderConfig()


# ---------- degradations ---------------------------------------------------

class DegradationConfig(BaseModel):
    """Per-task configuration. Extra keys are passed to the degradation
    constructor as kwargs (e.g. sigma_range, factor)."""

    model_config = {"extra": "allow"}
    weight: float = 1.0


# ---------- losses ---------------------------------------------------------

class LossConfig(BaseModel):
    name: str
    weight: float = 1.0
    config: dict[str, Any] = Field(default_factory=dict)
    apply_to_tasks: list[str] | None = None  # None == all tasks


_LOSS_PRESETS: dict[str, list[dict[str, Any]]] = {
    "minimal": [{"name": "l1_rgb", "weight": 1.0}],
    "standard": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "colorfulness", "weight": 0.3, "apply_to_tasks": ["colorize"]},
        {"name": "freq_l1", "weight": 0.2, "apply_to_tasks": ["sr_x2", "sr_x4", "deblur"]},
    ],
    "vivid": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "colorfulness", "weight": 2.0, "apply_to_tasks": ["colorize"]},
        {"name": "freq_l1", "weight": 0.2, "apply_to_tasks": ["sr_x2", "sr_x4", "deblur"]},
    ],
    "full": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "colorfulness", "weight": 0.3, "apply_to_tasks": ["colorize"]},
        {"name": "freq_l1", "weight": 0.2, "apply_to_tasks": ["sr_x2", "sr_x4", "deblur"]},
        {"name": "gan", "weight": 0.1, "config": {"gan_type": "hinge"},
         "apply_to_tasks": ["colorize", "sr_x2", "sr_x4"]},
    ],
}


def expand_loss_preset(name: str) -> list[LossConfig]:
    if name not in _LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {name!r}; have {list(_LOSS_PRESETS)}")
    return [LossConfig(**d) for d in _LOSS_PRESETS[name]]


# ---------- optimizer / scheduler -----------------------------------------

class OptimConfig(BaseModel):
    type: Literal["AdamW", "Adam"] = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    fused: bool = True


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "multistep", "constant"] = "cosine"
    warmup_steps: int = 2000
    total_steps: int = 100_000
    milestones: list[int] = Field(default_factory=list)
    gamma: float = 0.5


# ---------- training ------------------------------------------------------

class TrainConfig(BaseModel):
    total_steps: int = 100_000
    amp: Literal["bf16", "fp16", "fp32"] = "bf16"
    memory_format: Literal["channels_last", "contiguous"] = "channels_last"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    ema_decay: float = 0.999
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    preview_every_s: float = 10.0
    preview_history_every: int = 50
    ckpt_every_steps: int = 5000
    val_every_steps: int = 5000
    log_every_steps: int = 25


class ExportConfig(BaseModel):
    on_finish: bool = True
    opset: int = 17
    simplify: bool = True
    dynamic_hw: bool = False


class RunConfig(BaseModel):
    name: str = ""
    output_dir: str = ""
    seed: int = 0


class Config(BaseModel):
    run: RunConfig = RunConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig
    degradations: dict[str, DegradationConfig]
    losses: list[LossConfig]
    optim_g: OptimConfig = OptimConfig()
    optim_d: OptimConfig = OptimConfig(weight_decay=0.0)
    scheduler: SchedulerConfig = SchedulerConfig()
    train: TrainConfig = TrainConfig()
    export: ExportConfig = ExportConfig()


# ---------- YAML loader ---------------------------------------------------

class _LoaderWithPresets(yaml.SafeLoader):
    pass


def _preset_constructor(loader, node):
    name = loader.construct_scalar(node)
    return [d.copy() for d in _LOSS_PRESETS[name]]


_LoaderWithPresets.add_constructor("!preset", _preset_constructor)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _interpolate_date(value: str) -> str:
    while "${date:" in value:
        start = value.index("${date:")
        end = value.index("}", start)
        fmt = value[start + 7 : end]
        value = value[:start] + _dt.datetime.now().strftime(fmt) + value[end + 1 :]
    return value


def _load_yaml_chained(path: Path) -> dict:
    with path.open() as fh:
        d = yaml.load(fh, Loader=_LoaderWithPresets) or {}
    defaults = d.pop("defaults", None)
    if defaults:
        parent_path = (path.parent / defaults).resolve()
        parent = _load_yaml_chained(parent_path)
        d = deep_merge(parent, d)
    return d


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path)
    raw = _load_yaml_chained(path)
    if overrides:
        raw = deep_merge(raw, overrides)

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            return _interpolate_date(x)
        return x

    return Config.model_validate(walk(raw))
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_config.py -q   # 6 passed
git add src/refine/config.py tests/test_config.py
git commit -m "refine: pydantic config + YAML loader with task-aware losses

Adds DegradationConfig and apply_to_tasks on LossConfig. Presets:
minimal / standard / vivid / full. Otherwise mirrors v1's structure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Degradation registry + base class

**Files:**
- Create: `src/refine/data/degradations/registry.py`
- Create: `tests/test_degradation_registry.py`

- [ ] **Step 1: Test**

```python
import numpy as np
import pytest

from refine.data.degradations.registry import (
    DEGRADATION_REGISTRY,
    Degradation,
    build_degradation,
    register_degradation,
)


def test_registry_collects_decorated_class():
    @register_degradation("toy_test")
    class _Toy(Degradation):
        name = "toy_test"

        def __init__(self, weight: float = 1.0):
            super().__init__()
            self.weight = weight

        def degrade(self, rgb, rng):
            return rgb

    assert "toy_test" in DEGRADATION_REGISTRY
    d = build_degradation("toy_test", {"weight": 2.0})
    assert d.weight == 2.0
    assert d.degrade(np.zeros((4, 4, 3), dtype=np.float32), None).shape == (4, 4, 3)
    DEGRADATION_REGISTRY.pop("toy_test")


def test_build_unknown_raises():
    with pytest.raises(KeyError):
        build_degradation("nope", {})
```

- [ ] **Step 2: Implement `src/refine/data/degradations/registry.py`**

```python
"""Degradation registry. Each registered Degradation has a name and
a degrade() method that maps (H, W, 3) float32 RGB → degraded same-shape RGB."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Type

import numpy as np


class Degradation(ABC):
    name: str = ""
    task_id: int = -1  # assigned by config at startup

    @abstractmethod
    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        """rgb: (H, W, 3) float32 in [0, 1]. Returns same-shape degraded RGB."""


DEGRADATION_REGISTRY: dict[str, Type[Degradation]] = {}


def register_degradation(name: str):
    def deco(cls: Type[Degradation]):
        if name in DEGRADATION_REGISTRY:
            raise KeyError(f"degradation {name!r} already registered")
        cls.name = name
        DEGRADATION_REGISTRY[name] = cls
        return cls

    return deco


def build_degradation(name: str, cfg: dict[str, Any] | None = None) -> Degradation:
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"unknown degradation {name!r}; have {sorted(DEGRADATION_REGISTRY)}")
    cfg = dict(cfg or {})
    cfg.pop("weight", None)  # weight is for sampling, not constructor
    return DEGRADATION_REGISTRY[name](**cfg)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_degradation_registry.py -q   # 2 passed
git add src/refine/data/degradations/registry.py tests/test_degradation_registry.py
git commit -m "refine: degradation registry + base class

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Five degradation implementations

**Files:**
- Create: `src/refine/data/degradations/colorization.py`
- Create: `src/refine/data/degradations/denoise.py`
- Create: `src/refine/data/degradations/superres.py`
- Create: `src/refine/data/degradations/deblur.py`
- Create: `src/refine/data/degradations/jpeg.py`
- Create: `tests/test_degradations.py`

- [ ] **Step 1: Test `tests/test_degradations.py`**

```python
import random

import numpy as np
import pytest

from refine.data.degradations import colorization, denoise, deblur, jpeg, superres  # noqa: F401
from refine.data.degradations.registry import DEGRADATION_REGISTRY


@pytest.fixture
def rgb_in():
    rng = np.random.default_rng(0)
    return rng.random((48, 64, 3)).astype(np.float32)


@pytest.mark.parametrize("name,cfg", [
    ("colorize", {}),
    ("denoise",  {"sigma_range": [0.01, 0.05]}),
    ("sr_x2",    {"factor": 2}),
    ("sr_x4",    {"factor": 4}),
    ("deblur",   {"sigma_range": [1.0, 2.0], "motion_prob": 0.0}),
    ("jpeg",     {"quality_range": [40, 60]}),
])
def test_degradation_preserves_shape_and_dtype(name, cfg, rgb_in):
    d_cls = DEGRADATION_REGISTRY[name]
    d_cfg = dict(cfg); d_cfg.pop("weight", None)
    d = d_cls(**d_cfg)
    rng = random.Random(0)
    out = d.degrade(rgb_in.copy(), rng)
    assert out.shape == rgb_in.shape
    assert out.dtype == rgb_in.dtype
    assert out.min() >= 0.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5


def test_colorization_zeros_chroma(rgb_in):
    import cv2

    out = DEGRADATION_REGISTRY["colorize"]().degrade(rgb_in.copy(), random.Random(0))
    # Convert to LAB and check a, b ≈ 0
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    assert abs(lab[:, :, 1]).mean() < 1.0  # a ≈ 0
    assert abs(lab[:, :, 2]).mean() < 1.0  # b ≈ 0


def test_denoise_adds_noise(rgb_in):
    d = DEGRADATION_REGISTRY["denoise"](sigma_range=[0.03, 0.03])
    out = d.degrade(rgb_in.copy(), random.Random(0))
    # different from input
    assert (out - rgb_in).std() > 0.01


def test_sr_x4_actually_loses_detail(rgb_in):
    """SR degradation = downsample + upsample loses some high-frequency content."""
    d = DEGRADATION_REGISTRY["sr_x4"](factor=4)
    out = d.degrade(rgb_in.copy(), random.Random(0))
    # output is smoother than input (mean abs gradient is lower)
    grad_in = np.abs(np.diff(rgb_in, axis=0)).mean()
    grad_out = np.abs(np.diff(out, axis=0)).mean()
    assert grad_out < grad_in
```

- [ ] **Step 2: `src/refine/data/degradations/colorization.py`**

```python
"""Colorization: RGB → gray-as-RGB via LAB-L (a=b=0)."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


@register_degradation("colorize")
class ColorizationDegradation(Degradation):
    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        L = lab[:, :, :1]
        gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
        gray_rgb = cv2.cvtColor(cv2.cvtColor(gray_lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        return gray_rgb.astype(np.float32)
```

- [ ] **Step 3: `src/refine/data/degradations/denoise.py`**

```python
"""Denoising: add Gaussian noise (optionally + Poisson) with random sigma."""
from __future__ import annotations

import random

import numpy as np

from .registry import Degradation, register_degradation


@register_degradation("denoise")
class DenoiseDegradation(Degradation):
    def __init__(
        self,
        sigma_range: tuple[float, float] = (0.005, 0.05),
        poisson_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.sigma_min, self.sigma_max = float(sigma_range[0]), float(sigma_range[1])
        self.poisson_prob = float(poisson_prob)

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        sigma = rng.uniform(self.sigma_min, self.sigma_max)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        noise = np_rng.normal(0.0, sigma, rgb.shape).astype(np.float32)
        out = rgb + noise
        if rng.random() < self.poisson_prob:
            # cheap poisson-ish: scale-dependent gaussian
            poisson_noise = np_rng.normal(0.0, 0.01 * np.sqrt(np.clip(rgb, 1e-3, 1.0)),
                                          rgb.shape).astype(np.float32)
            out = out + poisson_noise
        return np.clip(out, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: `src/refine/data/degradations/superres.py`**

```python
"""Super-resolution degradation: bicubic down, then bicubic up. Same-resolution output."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


class _SRBase(Degradation):
    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = int(factor)

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (max(1, w // self.factor), max(1, h // self.factor)),
                           interpolation=cv2.INTER_CUBIC)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(up, 0.0, 1.0).astype(np.float32)


@register_degradation("sr_x2")
class SRx2(_SRBase):
    def __init__(self, factor: int = 2):
        super().__init__(factor=factor)


@register_degradation("sr_x4")
class SRx4(_SRBase):
    def __init__(self, factor: int = 4):
        super().__init__(factor=factor)
```

- [ ] **Step 5: `src/refine/data/degradations/deblur.py`**

```python
"""Deblur degradation: Gaussian blur (optional motion blur)."""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    k /= k.sum() + 1e-8
    return k


@register_degradation("deblur")
class DeblurDegradation(Degradation):
    def __init__(
        self,
        sigma_range: tuple[float, float] = (1.0, 3.0),
        motion_prob: float = 0.2,
        motion_size_range: tuple[int, int] = (7, 21),
    ) -> None:
        super().__init__()
        self.sigma_min, self.sigma_max = float(sigma_range[0]), float(sigma_range[1])
        self.motion_prob = float(motion_prob)
        self.motion_min, self.motion_max = int(motion_size_range[0]), int(motion_size_range[1])

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        if rng.random() < self.motion_prob:
            size = rng.randint(self.motion_min, self.motion_max)
            if size % 2 == 0:
                size += 1
            angle = rng.uniform(0.0, 180.0)
            kernel = _motion_kernel(size, angle)
            out = cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REFLECT)
        else:
            sigma = rng.uniform(self.sigma_min, self.sigma_max)
            ksize = max(3, int(2 * math.ceil(2 * sigma) + 1))
            out = cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                   borderType=cv2.BORDER_REFLECT)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 6: `src/refine/data/degradations/jpeg.py`**

```python
"""JPEG-restore degradation: encode then decode at random quality."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


@register_degradation("jpeg")
class JpegDegradation(Degradation):
    def __init__(self, quality_range: tuple[int, int] = (20, 70)) -> None:
        super().__init__()
        self.qmin, self.qmax = int(quality_range[0]), int(quality_range[1])

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        quality = rng.randint(self.qmin, self.qmax)
        bgr_uint8 = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * 255.0).round().clip(0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", bgr_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return rgb
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(decoded_rgb, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 7: Run + commit**

```bash
uv run pytest tests/test_degradations.py -q   # 9 passed (6 param + 3 specific)
git add src/refine/data/degradations tests/test_degradations.py
git commit -m "refine: five degradation implementations

colorize, denoise, sr_x2, sr_x4, deblur, jpeg — each one file, all
registered via @register_degradation. Same-resolution RGB out for
every degradation matches the model's input contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Clean dataset (lifted from v1)

**Files:**
- Create: `src/refine/data/transforms.py`
- Create: `src/refine/data/dataset.py`
- Create: `tests/test_dataset.py`

Lifted from v1 with one change: returns clean RGB only (no LAB derivation in dataset).

- [ ] **Step 1: Test**

```python
from pathlib import Path

import cv2
import numpy as np
import torch

from refine.data.dataset import MANIFEST_NAME, RecursiveImageDataset, build_manifest


def test_build_manifest_finds_all_images(tmp_image_dir):
    assert len(build_manifest(tmp_image_dir)) == 6


def test_manifest_is_cached(tmp_image_dir):
    build_manifest(tmp_image_dir)
    assert (tmp_image_dir / MANIFEST_NAME).exists()


def test_dataset_returns_clean_rgb(tmp_image_dir):
    ds = RecursiveImageDataset(tmp_image_dir, target_size=32, augment_hflip=False)
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 32, 32)
    assert sample.dtype == torch.float32
    assert sample.min() >= 0.0 and sample.max() <= 1.0


def test_skip_too_small(tmp_path):
    cv2.imwrite(str(tmp_path / "ok.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "tiny.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    ds = RecursiveImageDataset(tmp_path, target_size=32, min_side=32, augment_hflip=False)
    assert len(ds) == 1


def test_deterministic_split(tmp_image_dir):
    a = RecursiveImageDataset(tmp_image_dir, target_size=32, val_fraction=0.34, split="val",
                              augment_hflip=False)
    b = RecursiveImageDataset(tmp_image_dir, target_size=32, val_fraction=0.34, split="val",
                              augment_hflip=False)
    assert [str(p) for p in a._paths] == [str(p) for p in b._paths]
```

- [ ] **Step 2: Implement `src/refine/data/transforms.py`**

```python
"""Tiny transforms applied numpy-side, before tensorization."""
from __future__ import annotations

import random

import numpy as np


def random_crop(rgb: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {(h, w)} smaller than crop {size}")
    top = rng.randint(0, h - size)
    left = rng.randint(0, w - size)
    return rgb[top : top + size, left : left + size]


def hflip(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, ::-1])
```

- [ ] **Step 3: Implement `src/refine/data/dataset.py`**

```python
"""Recursive image dataset with manifest cache and deterministic train/val split.

Returns *clean* (3, H, W) float32 RGB. Degradation lives outside the dataset
(see refine.data.multitask.MultiTaskWrapper).
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import hflip, random_crop

MANIFEST_NAME = ".refine-manifest.txt"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def build_manifest(root: Path, *, force: bool = False) -> list[Path]:
    root = Path(root)
    mf = _manifest_path(root)
    if mf.exists() and not force:
        try:
            lines = mf.read_text().splitlines()
            mtime = float(lines[0])
            if abs(mtime - root.stat().st_mtime) < 1.0:
                return [root / line for line in lines[1:]]
        except Exception:
            pass
    out = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in _EXTS and p.is_file()]
    try:
        mf.write_text(f"{root.stat().st_mtime}\n" + "\n".join(str(p.relative_to(root)) for p in out))
    except OSError:
        pass
    return out


def _hash_to_unit(path: Path) -> float:
    return int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class RecursiveImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        target_size: int,
        min_side: int | None = None,
        val_fraction: float = 0.0,
        split: Literal["train", "val", "all"] = "all",
        augment_hflip: bool = True,
        augment_rotate90: bool = False,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.target_size = target_size
        self.min_side = min_side if min_side is not None else target_size
        self.augment_hflip = augment_hflip
        self.augment_rotate90 = augment_rotate90
        self._seed = seed

        from PIL import Image  # header-only read, fast on remote mounts

        all_paths = build_manifest(self.root)
        kept: list[Path] = []
        for p in all_paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                if h < self.min_side or w < self.min_side:
                    continue
                kept.append(p)
            except Exception:
                continue
        if val_fraction > 0 and split != "all":
            wanted = "val" if split == "val" else "train"
            kept = [p for p in kept if ((_hash_to_unit(p) < val_fraction) == (wanted == "val"))]
        self._paths = kept

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self._paths[idx]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        rng = random.Random((self._seed * 1_000_003) ^ idx)
        if self.augment_hflip and rng.random() < 0.5:
            rgb = hflip(rgb)
        if self.augment_rotate90 and rng.random() < 0.5:
            rgb = np.ascontiguousarray(np.rot90(rgb, k=rng.choice([1, 2, 3])))

        rgb = random_crop(rgb, self.target_size, rng)
        rgb_f32 = rgb.astype(np.float32) / 255.0
        return torch.from_numpy(rgb_f32.transpose(2, 0, 1)).contiguous()
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_dataset.py -q   # 5 passed
git add src/refine/data/dataset.py src/refine/data/transforms.py tests/test_dataset.py
git commit -m "refine: clean RGB dataset (lifted from v1, manifest .refine-manifest.txt)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: MultiTaskWrapper

**Files:**
- Create: `src/refine/data/multitask.py`
- Create: `tests/test_multitask.py`

- [ ] **Step 1: Test**

```python
import random

import numpy as np
import torch

from refine.data.degradations import colorization, denoise, superres  # noqa: F401
from refine.data.degradations.registry import build_degradation
from refine.data.multitask import MultiTaskWrapper, collate_multitask


class _DummyCleanDS:
    def __len__(self): return 8
    def __getitem__(self, idx):
        return torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(idx))


def test_wrapper_picks_tasks_via_weights():
    ds = _DummyCleanDS()
    degs = [build_degradation("colorize"), build_degradation("denoise")]
    # set task ids
    degs[0].task_id = 0; degs[1].task_id = 1
    wrap = MultiTaskWrapper(ds, degs, weights=[0.5, 0.5], seed=0)
    counts = {0: 0, 1: 0}
    for i in range(200):
        s = wrap[i]
        counts[int(s["task_id"])] += 1
    # both tasks selected at least 30 % of the time
    assert counts[0] >= 60 and counts[1] >= 60


def test_wrapper_sample_shapes():
    ds = _DummyCleanDS()
    deg = build_degradation("sr_x2"); deg.task_id = 0
    wrap = MultiTaskWrapper(ds, [deg], weights=[1.0], seed=0)
    s = wrap[0]
    assert s["clean"].shape == (3, 32, 32)
    assert s["degraded"].shape == (3, 32, 32)
    assert s["task_id"].item() == 0
    assert s["task_name"] == "sr_x2"


def test_collate_stacks():
    ds = _DummyCleanDS()
    deg = build_degradation("denoise"); deg.task_id = 0
    wrap = MultiTaskWrapper(ds, [deg], weights=[1.0], seed=0)
    batch = collate_multitask([wrap[i] for i in range(4)])
    assert batch["clean"].shape == (4, 3, 32, 32)
    assert batch["degraded"].shape == (4, 3, 32, 32)
    assert batch["task_id"].shape == (4,)
    assert batch["task_name"] == ["denoise"] * 4
```

- [ ] **Step 2: Implement `src/refine/data/multitask.py`**

```python
"""MultiTaskWrapper: per-sample task picker on top of a clean-image dataset."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations.registry import Degradation


class MultiTaskWrapper(Dataset):
    def __init__(self, clean_ds: Dataset, degradations: list[Degradation],
                 weights: list[float], *, seed: int = 0) -> None:
        if len(degradations) == 0:
            raise ValueError("at least one degradation required")
        if len(degradations) != len(weights):
            raise ValueError("degradations/weights length mismatch")
        self.clean = clean_ds
        self.degs = degradations
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights sum must be > 0")
        self.cdf = np.cumsum(np.asarray([w / total for w in weights], dtype=np.float64))
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int) -> dict:
        clean = self.clean[idx]  # (3, H, W) torch.float32, [0, 1]
        rng = random.Random((self.seed * 1_000_003) ^ idx)
        task_idx = int(np.searchsorted(self.cdf, rng.random()))
        if task_idx >= len(self.degs):
            task_idx = len(self.degs) - 1
        deg = self.degs[task_idx]
        rgb_np = clean.permute(1, 2, 0).numpy()
        degraded_np = deg.degrade(rgb_np, rng)
        return {
            "clean": clean,
            "degraded": torch.from_numpy(degraded_np.transpose(2, 0, 1)).contiguous(),
            "task_id": torch.tensor(deg.task_id, dtype=torch.long),
            "task_name": deg.name,
        }


def collate_multitask(batch: list[dict]) -> dict:
    return {
        "clean": torch.stack([b["clean"] for b in batch]),
        "degraded": torch.stack([b["degraded"] for b in batch]),
        "task_id": torch.stack([b["task_id"] for b in batch]),
        "task_name": [b["task_name"] for b in batch],
    }
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_multitask.py -q   # 3 passed
git add src/refine/data/multitask.py tests/test_multitask.py
git commit -m "refine: MultiTaskWrapper for per-sample task picking

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Task embedding module

**Files:**
- Create: `src/refine/models/task_embed.py`
- Create: `tests/test_task_embed.py`

- [ ] **Step 1: Test**

```python
import torch

from refine.models.task_embed import TaskEmbed


def test_task_embed_shape():
    m = TaskEmbed(num_tasks=6, dim=128)
    task = torch.tensor([0, 1, 5, 2], dtype=torch.long)
    out = m(task)
    assert out.shape == (4, 128)


def test_task_embed_distinguishes_tasks():
    m = TaskEmbed(num_tasks=6, dim=64)
    a = m(torch.tensor([0])).detach()
    b = m(torch.tensor([1])).detach()
    assert (a - b).abs().sum() > 0
```

- [ ] **Step 2: Implement `src/refine/models/task_embed.py`**

```python
"""Task embedding + MLP, used to condition NAFBlocks (FiLM) and the
bottleneck transformer (AdaLN)."""
from __future__ import annotations

import torch
from torch import nn


class TaskEmbed(nn.Module):
    def __init__(self, *, num_tasks: int, dim: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_tasks, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(task))
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_task_embed.py -q
git add src/refine/models/task_embed.py tests/test_task_embed.py
git commit -m "refine: task embedding (nn.Embedding + 2-layer MLP)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: NAFBlock with FiLM conditioning

**Files:**
- Create: `src/refine/models/nafblock.py`
- Create: `tests/test_nafblock.py`

- [ ] **Step 1: Test**

```python
import torch

from refine.models.nafblock import NAFBlock


def test_nafblock_shape():
    blk = NAFBlock(c=16, task_dim=32)
    x = torch.randn(2, 16, 8, 8)
    t = torch.randn(2, 32)
    assert blk(x, t).shape == x.shape


def test_film_conditions_output():
    blk = NAFBlock(c=16, task_dim=32)
    x = torch.randn(1, 16, 8, 8)
    t1 = torch.randn(1, 32)
    t2 = torch.randn(1, 32) * 3.0
    assert (blk(x, t1) - blk(x, t2)).abs().sum() > 0


def test_residual_path():
    blk = NAFBlock(c=8, task_dim=16)
    x = torch.randn(1, 8, 6, 6, requires_grad=True)
    t = torch.randn(1, 16)
    blk(x, t).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
```

- [ ] **Step 2: Implement (see full code in plan source — NAFBlock with FiLM, channel-wise LayerNorm, SimpleGate, simple channel attention)**

The implementation is ~80 lines. Key contract: `forward(x, task_vec)` where x is (B, c, H, W) and task_vec is (B, task_dim). FiLM γ is shifted by +1 so γ≈0 at init gives identity. Full source listed in the plan appendix below.

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_nafblock.py -q
git add src/refine/models/nafblock.py tests/test_nafblock.py
git commit -m "refine: NAFBlock with FiLM conditioning

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### NAFBlock implementation (apply in Step 2 above)

```python
"""NAFBlock from 'Simple Baselines for Image Restoration' (Chen et al. ECCV'22),
with FiLM conditioning on a task vector."""
from __future__ import annotations

import torch
from torch import nn


class _ChannelLayerNorm(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class _SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class _SimpleChannelAttention(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c: int, *, task_dim: int, expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        self.film = nn.Linear(task_dim, 2 * c)
        self.norm1 = _ChannelLayerNorm(c)
        self.conv1 = nn.Conv2d(c, c * expand * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(c * expand * 2, c * expand * 2, kernel_size=3, padding=1,
                                groups=c * expand * 2)
        self.gate1 = _SimpleGate()
        self.sca = _SimpleChannelAttention(c * expand)
        self.conv2 = nn.Conv2d(c * expand, c, kernel_size=1)
        self.norm2 = _ChannelLayerNorm(c)
        self.conv3 = nn.Conv2d(c, c * ffn_expand * 2, kernel_size=1)
        self.gate2 = _SimpleGate()
        self.conv4 = nn.Conv2d(c * ffn_expand, c, kernel_size=1)

    def forward(self, x: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(task_vec).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = self.norm1(x)
        h = (1.0 + gamma) * h + beta
        h = self.conv1(h)
        h = self.dwconv(h)
        h = self.gate1(h)
        h = self.sca(h)
        h = self.conv2(h)
        x = x + h
        h = self.norm2(x)
        h = self.conv3(h)
        h = self.gate2(h)
        h = self.conv4(h)
        return x + h
```

---

## Task 12: Transformer block with AdaLN

**Files:**
- Create: `src/refine/models/transformer_block.py`
- Create: `tests/test_transformer_block.py`

- [ ] **Step 1: Test**

```python
import torch

from refine.models.transformer_block import TransformerBlock


def test_transformer_block_shape():
    blk = TransformerBlock(c=64, task_dim=32, num_heads=4, ffn_dim=128)
    x = torch.randn(2, 64, 8, 8)
    t = torch.randn(2, 32)
    assert blk(x, t).shape == x.shape


def test_adaln_conditions_output():
    blk = TransformerBlock(c=64, task_dim=32, num_heads=4, ffn_dim=128)
    x = torch.randn(1, 64, 8, 8)
    t1 = torch.zeros(1, 32)
    t2 = torch.ones(1, 32) * 2.0
    assert (blk(x, t1) - blk(x, t2)).abs().sum() > 0
```

- [ ] **Step 2: Implement `src/refine/models/transformer_block.py`**

```python
"""Transformer block (MHSA + FFN) with AdaLN conditioning on a task vector.
Adapted from DiT (Peebles & Xie, 2023)."""
from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, *, c: int, task_dim: int, num_heads: int = 8, ffn_dim: int = 256) -> None:
        super().__init__()
        self.adaln1 = nn.Linear(task_dim, 2 * c)
        self.adaln2 = nn.Linear(task_dim, 2 * c)
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, c),
        )

    def forward(self, x: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        gamma1, beta1 = self.adaln1(task_vec).chunk(2, dim=-1)
        gamma2, beta2 = self.adaln2(task_vec).chunk(2, dim=-1)
        gamma1 = gamma1.unsqueeze(1); beta1 = beta1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1); beta2 = beta2.unsqueeze(1)
        h_mod = self.norm1(seq) * (1.0 + gamma1) + beta1
        attn_out, _ = self.attn(h_mod, h_mod, h_mod, need_weights=False)
        seq = seq + attn_out
        h_mod = self.norm2(seq) * (1.0 + gamma2) + beta2
        seq = seq + self.ffn(h_mod)
        return seq.transpose(1, 2).reshape(b, c, h, w)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_transformer_block.py -q
git add src/refine/models/transformer_block.py tests/test_transformer_block.py
git commit -m "refine: transformer block with AdaLN task conditioning

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Model registry + NAFNet multi-task

**Files:**
- Create: `src/refine/models/registry.py`
- Create: `src/refine/models/nafnet.py`
- Modify: `src/refine/models/__init__.py`
- Create: `tests/test_nafnet.py`

- [ ] **Step 1: Test**

```python
import torch

from refine.config import ModelConfig
from refine.models import build_model


def test_nafnet_tiny_forward_shape():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_tasks=3)
    rgb = torch.rand(2, 3, 32, 32)
    task = torch.tensor([0, 2], dtype=torch.long)
    out = m(rgb, task)
    assert out.shape == rgb.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_nafnet_at_init_is_near_identity():
    """Residual + zero-init head: untrained model passes input through."""
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_tasks=3)
    m.train(False)
    rgb = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = m(rgb, torch.tensor([0]))
    # Zero-init head means delta_lab_n = 0; output ≈ input within LAB rounding
    assert (out - rgb).abs().mean() < 0.05


def test_nafnet_backward_flows():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_tasks=3)
    rgb = torch.rand(1, 3, 32, 32)
    out = m(rgb, torch.tensor([1]))
    out.pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
```

- [ ] **Step 2: Implement `src/refine/models/registry.py`**

```python
"""Model registry: future architectures plug in via @register_model with
the same forward contract (rgb, task) -> rgb."""
from __future__ import annotations

from typing import Type

from torch import nn

from refine.config import ModelConfig

MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    def deco(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            raise KeyError(f"model {name!r} already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


def build_model(cfg: ModelConfig, *, num_tasks: int) -> nn.Module:
    if cfg.type not in MODEL_REGISTRY:
        raise KeyError(f"unknown model type {cfg.type!r}; have {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[cfg.type](cfg, num_tasks=num_tasks)
```

- [ ] **Step 3: Implement `src/refine/models/nafnet.py`**

```python
"""NAFNet multi-task model."""
from __future__ import annotations

import torch
from torch import nn

from refine.config import ModelConfig
from .color import LabToRgb, RgbToLab
from .nafblock import NAFBlock
from .registry import register_model
from .task_embed import TaskEmbed
from .transformer_block import TransformerBlock


_SIZE_PRESETS: dict[str, dict] = {
    "tiny": {"nf": 32, "enc_depths": [2, 2, 2, 2], "bottle_blocks": 2, "hidden_dim": 256},
    "large": {"nf": 64, "enc_depths": [2, 2, 4, 8], "bottle_blocks": 4, "hidden_dim": 384},
}


def _resolve(cfg: ModelConfig) -> dict:
    preset = _SIZE_PRESETS[cfg.size]
    return {
        "nf": cfg.nf if cfg.nf is not None else preset["nf"],
        "enc_depths": cfg.enc_depths if cfg.enc_depths is not None else preset["enc_depths"],
        "bottle_blocks": cfg.bottle_blocks if cfg.bottle_blocks is not None else preset["bottle_blocks"],
        "hidden_dim": cfg.hidden_dim if cfg.hidden_dim is not None else preset["hidden_dim"],
        "task_dim": cfg.task_embed_dim,
    }


@register_model("nafnet")
class NAFNetMultiTask(nn.Module):
    def __init__(self, cfg: ModelConfig, *, num_tasks: int) -> None:
        super().__init__()
        p = _resolve(cfg)
        nf = p["nf"]; depths = p["enc_depths"]; bottle_n = p["bottle_blocks"]
        hidden = p["hidden_dim"]; task_dim = p["task_dim"]
        assert len(depths) == 4

        self.rgb_to_lab = RgbToLab()
        self.lab_to_rgb = LabToRgb()
        self.task_embed = TaskEmbed(num_tasks=num_tasks, dim=task_dim)

        self.stem = nn.Conv2d(3, nf, kernel_size=3, padding=1)

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = nf
        enc_channels: list[int] = []
        for n in depths:
            self.enc_stages.append(nn.ModuleList([NAFBlock(ch, task_dim=task_dim) for _ in range(n)]))
            enc_channels.append(ch)
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        self.bottle_in = nn.Conv2d(ch, hidden, kernel_size=1)
        self.bottle = nn.ModuleList([
            TransformerBlock(c=hidden, task_dim=task_dim, num_heads=8, ffn_dim=hidden * 2)
            for _ in range(bottle_n)
        ])
        self.bottle_out = nn.Conv2d(hidden, ch, kernel_size=1)

        self.ups = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for n, skip_c in zip(reversed(depths), reversed(enc_channels)):
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, skip_c * 4, kernel_size=1),
                nn.PixelShuffle(2),
            ))
            ch = skip_c
            self.skip_proj.append(nn.Conv2d(ch * 2, ch, kernel_size=1))
            self.dec_stages.append(nn.ModuleList([NAFBlock(ch, task_dim=task_dim) for _ in range(n)]))

        self.head = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, rgb: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        lab_n = self.rgb_to_lab(rgb)
        task_vec = self.task_embed(task)

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

        delta_lab_n = self.head(x)
        return self.lab_to_rgb(lab_n + delta_lab_n)
```

- [ ] **Step 4: Update `src/refine/models/__init__.py`**

```python
"""Refine models. Importing this module registers the NAFNet backbone."""
from . import nafnet as _nafnet  # noqa: F401
from .registry import MODEL_REGISTRY, build_model, register_model

__all__ = ["MODEL_REGISTRY", "build_model", "register_model"]
```

- [ ] **Step 5: Run + commit**

```bash
uv run pytest tests/test_nafnet.py -q
git add src/refine/models tests/test_nafnet.py
git commit -m "refine: NAFNet multi-task backbone + model registry

Encoder 4 stages (NAFBlocks + strided conv) -> bottleneck transformer
with AdaLN -> decoder 4 stages (pixel-shuffle + skip + NAFBlocks) ->
head (zero-init weights for identity-at-init). LAB conversion via the
frozen fp32 modules. @register_model('nafnet'); future architectures
plug in the same way.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Loss registry + LossContext + pixel losses

**Files:**
- Create: `src/refine/losses/registry.py`
- Create: `src/refine/losses/pixel.py`
- Create: `tests/test_pixel_losses.py`

- [ ] **Step 1: Test**

```python
import torch

from refine.losses.pixel import CharbonnierRgbLoss, L1RgbLoss
from refine.losses.registry import LossContext


def _ctx(pred=None, clean=None):
    z = torch.zeros(2, 3, 4, 4)
    return LossContext(
        pred_rgb=pred if pred is not None else z.clone(),
        clean_rgb=clean if clean is not None else z.clone(),
        degraded_rgb=z.clone(),
        task_ids=torch.tensor([0, 0]),
        task_names=["colorize", "colorize"],
    )


def test_l1_zero_when_equal():
    assert L1RgbLoss()(_ctx()).item() == 0.0


def test_l1_positive_when_unequal():
    pred = torch.ones(2, 3, 4, 4)
    assert L1RgbLoss()(_ctx(pred=pred)).item() == 1.0


def test_charbonnier_grad():
    pred = torch.randn(2, 3, 4, 4, requires_grad=True)
    CharbonnierRgbLoss()(_ctx(pred=pred)).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
```

- [ ] **Step 2: `src/refine/losses/registry.py`**

```python
"""Loss registry + LossContext + build_loss factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import torch
from torch import nn


@dataclass
class LossContext:
    pred_rgb: torch.Tensor          # (B, 3, H, W) model output
    clean_rgb: torch.Tensor         # (B, 3, H, W) ground truth
    degraded_rgb: torch.Tensor      # (B, 3, H, W) model input
    task_ids: torch.Tensor          # (B,) long
    task_names: list[str]           # length B
    discriminator: nn.Module | None = None


class RestorationLoss(nn.Module):
    name: str = ""

    def forward(self, ctx: LossContext) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


LOSS_REGISTRY: dict[str, Type[RestorationLoss]] = {}


def register_loss(name: str):
    def deco(cls: Type[RestorationLoss]):
        if name in LOSS_REGISTRY:
            raise KeyError(f"loss {name!r} already registered")
        cls.name = name
        LOSS_REGISTRY[name] = cls
        return cls

    return deco


def build_loss(name: str, cfg: dict[str, Any] | None = None) -> RestorationLoss:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"unknown loss {name!r}; have {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**(cfg or {}))
```

- [ ] **Step 3: `src/refine/losses/pixel.py`**

```python
"""Pixel-space losses (operate on pred_rgb vs clean_rgb)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("l1_rgb")
class L1RgbLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.l1_loss(ctx.pred_rgb, ctx.clean_rgb)


@register_loss("l2_rgb")
class L2RgbLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.mse_loss(ctx.pred_rgb, ctx.clean_rgb)


@register_loss("charbonnier_rgb")
class CharbonnierRgbLoss(RestorationLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        diff2 = (ctx.pred_rgb - ctx.clean_rgb) ** 2
        return torch.sqrt(diff2 + self.eps2).mean()
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_pixel_losses.py -q
git add src/refine/losses/registry.py src/refine/losses/pixel.py tests/test_pixel_losses.py
git commit -m "refine: loss registry + LossContext + RGB pixel losses

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Perceptual loss

**Files:**
- Create: `src/refine/losses/perceptual.py`
- Create: `tests/test_perceptual.py`

Lifted from v1 with fp32 dispatch already baked in. Operates on RGB (no change needed; v1's was already RGB).

- [ ] **Step 1: Test**

```python
import torch
from torch import nn

from refine.losses.perceptual import VGG16BNPerceptualLoss
from refine.losses.registry import LossContext


def test_perceptual_loss_grad_flows():
    """Stubbed VGG to avoid downloading weights in CI."""
    loss = VGG16BNPerceptualLoss.__new__(VGG16BNPerceptualLoss)
    nn.Module.__init__(loss)
    stub = nn.ModuleDict({
        "conv1_1": nn.Conv2d(3, 4, 3, padding=1),
        "conv2_1": nn.Conv2d(4, 8, 3, padding=1),
        "conv3_1": nn.Conv2d(8, 16, 3, padding=1),
    })
    loss._stages = stub
    loss._weights = {"conv1_1": 1.0, "conv2_1": 1.0, "conv3_1": 1.0}
    loss._criterion = nn.L1Loss()
    loss.style_weight = 0.0
    loss._input_norm = True
    loss.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    loss.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    pred = torch.randn(1, 3, 16, 16, requires_grad=True)
    gt = torch.randn(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=pred, clean_rgb=gt, degraded_rgb=z,
                      task_ids=torch.tensor([0]), task_names=["x"])
    loss(ctx).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
```

- [ ] **Step 2: Implement `src/refine/losses/perceptual.py`**

(Lifted from v1's `coliraz/losses/perceptual.py`. The forward signature uses `ctx.pred_rgb` / `ctx.clean_rgb` instead of `pred_rgb` / `gt_rgb`.)

```python
"""VGG16-BN perceptual loss with lazy weight load and fp32 dispatch."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from .registry import LossContext, RestorationLoss, register_loss

_LAYER_INDICES = {
    "conv1_1": 0, "conv1_2": 3,
    "conv2_1": 7, "conv2_2": 10,
    "conv3_1": 14, "conv3_2": 17, "conv3_3": 20,
    "conv4_1": 24, "conv4_2": 27, "conv4_3": 30,
    "conv5_1": 34, "conv5_2": 37, "conv5_3": 40,
}


def _gram(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    f = x.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


@register_loss("perceptual_vgg16bn")
class VGG16BNPerceptualLoss(RestorationLoss):
    def __init__(self, layer_weights: Mapping[str, float] | None = None,
                 criterion: str = "l1", style_weight: float = 0.0,
                 use_input_norm: bool = True) -> None:
        super().__init__()
        from torchvision.models import VGG16_BN_Weights, vgg16_bn

        if layer_weights is None:
            layer_weights = {"conv1_1": 0.0625, "conv2_1": 0.125, "conv3_1": 0.25,
                             "conv4_1": 0.5, "conv5_1": 1.0}
        self._weights = dict(layer_weights)
        self.style_weight = float(style_weight)
        self._input_norm = bool(use_input_norm)

        feats = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        feats.train(False)
        for p in feats.parameters():
            p.requires_grad_(False)

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        last = 0
        for name in sorted(self._weights, key=lambda k: _LAYER_INDICES[k]):
            idx = _LAYER_INDICES[name]
            stages[name] = nn.Sequential(*list(feats[last : idx + 1]))
            last = idx + 1
        self._stages = nn.ModuleDict(stages)
        self._criterion = nn.L1Loss() if criterion == "l1" else nn.MSELoss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._input_norm:
            x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        out: dict[str, torch.Tensor] = {}
        for name, stage in self._stages.items():
            x = stage(x)
            out[name] = x
        return out

    def forward(self, ctx: LossContext) -> torch.Tensor:
        device_type = ctx.pred_rgb.device.type
        with torch.amp.autocast(device_type, enabled=False):
            pred_rgb = ctx.pred_rgb.float()
            gt_rgb = ctx.clean_rgb.float()
            pred_f = self._features(pred_rgb)
            with torch.no_grad():
                gt_f = self._features(gt_rgb)
            perc: torch.Tensor | float = 0.0
            for name, w in self._weights.items():
                perc = perc + w * self._criterion(pred_f[name], gt_f[name].detach())
            if self.style_weight > 0:
                sty: torch.Tensor | float = 0.0
                for name, w in self._weights.items():
                    sty = sty + w * self._criterion(_gram(pred_f[name]), _gram(gt_f[name].detach()))
                perc = perc + self.style_weight * sty
        return perc
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_perceptual.py -q
git add src/refine/losses/perceptual.py tests/test_perceptual.py
git commit -m "refine: VGG16-BN perceptual loss (lifted from v1, fp32 dispatch)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: GAN loss + UNet discriminator + colorfulness

**Files:**
- Create: `src/refine/losses/gan.py`
- Create: `src/refine/losses/colorfulness.py`
- Create: `src/refine/models/discriminator.py`
- Create: `tests/test_gan_colorfulness.py`

GAN and colorfulness lifted from v1; discriminator lifted.

- [ ] **Step 1: Test**

```python
import torch
from torch import nn

from refine.losses.colorfulness import ColorfulnessLoss
from refine.losses.gan import GeneratorGANLoss, discriminator_loss
from refine.losses.registry import LossContext
from refine.models.discriminator import UNetDiscriminator


def _ctx(disc=None):
    rgb = torch.rand(2, 3, 16, 16, requires_grad=True)
    z = torch.zeros(2, 3, 16, 16)
    return LossContext(pred_rgb=rgb, clean_rgb=z, degraded_rgb=z,
                       task_ids=torch.tensor([0, 0]), task_names=["x", "x"],
                       discriminator=disc), rgb


def test_unet_discriminator_shape():
    d = UNetDiscriminator(nf=8)
    assert d(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_gen_gan_grad():
    disc = UNetDiscriminator(nf=8)
    loss = GeneratorGANLoss(gan_type="hinge")
    ctx, rgb = _ctx(disc=disc)
    loss(ctx).backward()
    assert rgb.grad is not None


def test_disc_loss_scalar():
    disc = UNetDiscriminator(nf=8)
    assert discriminator_loss(disc, torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16),
                              gan_type="hinge").dim() == 0


def test_colorfulness_grad():
    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    z = torch.zeros(1, 3, 4, 4)
    ctx = LossContext(pred_rgb=rgb, clean_rgb=z, degraded_rgb=z,
                     task_ids=torch.tensor([0]), task_names=["colorize"])
    ColorfulnessLoss()(ctx).backward()
    assert rgb.grad is not None
```

- [ ] **Step 2: `src/refine/models/discriminator.py`**

(Lifted from v1 — same UNet discriminator with spectral norm.)

```python
"""UNet-style image discriminator with per-pixel logits."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _down(in_c, out_c):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(in_c, out_c):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        spectral_norm(nn.Conv2d(in_c, out_c, 3, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )


class UNetDiscriminator(nn.Module):
    def __init__(self, *, in_ch: int = 3, nf: int = 64) -> None:
        super().__init__()
        self.d1 = _down(in_ch, nf)
        self.d2 = _down(nf, nf * 2)
        self.d3 = _down(nf * 2, nf * 4)
        self.d4 = _down(nf * 4, nf * 8)
        self.u3 = _up(nf * 8, nf * 4)
        self.u2 = _up(nf * 4 + nf * 4, nf * 2)
        self.u1 = _up(nf * 2 + nf * 2, nf)
        self.out = nn.Conv2d(nf + nf, 1, kernel_size=3, padding=1)
        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        u3 = self.u3(d4)
        u2 = self.u2(torch.cat([u3, d3], dim=1))
        u1 = self.u1(torch.cat([u2, d2], dim=1))
        return self.up_final(self.out(torch.cat([u1, d1], dim=1)))
```

- [ ] **Step 3: `src/refine/losses/gan.py`**

```python
"""Generator/discriminator GAN losses (vanilla, lsgan, hinge)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .registry import LossContext, RestorationLoss, register_loss


def _g_loss(logits, gan_type):
    if gan_type == "vanilla":
        return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    if gan_type == "lsgan":
        return F.mse_loss(logits, torch.ones_like(logits))
    if gan_type == "hinge":
        return -logits.mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


def _d_loss(real, fake, gan_type):
    if gan_type == "vanilla":
        return (F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
                F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
    if gan_type == "lsgan":
        return F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake))
    if gan_type == "hinge":
        return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


@register_loss("gan")
class GeneratorGANLoss(RestorationLoss):
    def __init__(self, gan_type: str = "hinge", discriminator: dict | None = None) -> None:
        super().__init__()
        self.gan_type = gan_type
        self._disc_cfg = discriminator or {"type": "unet", "nf": 64}

    @property
    def disc_config(self) -> dict:
        return self._disc_cfg

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.discriminator is None:
            raise RuntimeError("GeneratorGANLoss requires ctx.discriminator")
        return _g_loss(ctx.discriminator(ctx.pred_rgb), self.gan_type)


def discriminator_loss(disc, real_rgb, fake_rgb, gan_type):
    return _d_loss(disc(real_rgb), disc(fake_rgb.detach()), gan_type)
```

- [ ] **Step 4: `src/refine/losses/colorfulness.py`**

```python
"""Negated Hasler-Susstrunk colorfulness metric."""
from __future__ import annotations

import torch

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("colorfulness")
class ColorfulnessLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        rgb = ctx.pred_rgb.clamp(0, 1)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        sigma = torch.sqrt(rg.var(dim=(1, 2)) + yb.var(dim=(1, 2)) + 1e-8)
        mu = torch.sqrt(rg.mean(dim=(1, 2)) ** 2 + yb.mean(dim=(1, 2)) ** 2 + 1e-8)
        return -(sigma + 0.3 * mu).mean()
```

- [ ] **Step 5: Run + commit**

```bash
uv run pytest tests/test_gan_colorfulness.py -q
git add src/refine/losses/gan.py src/refine/losses/colorfulness.py \
        src/refine/models/discriminator.py tests/test_gan_colorfulness.py
git commit -m "refine: GAN losses + UNet discriminator + colorfulness (lifted from v1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Frequency-domain L1 loss

**Files:**
- Create: `src/refine/losses/freq.py`
- Create: `tests/test_freq_loss.py`

New: L1 between log-magnitude spectra of pred and clean RGB. Pushes for matching high-frequency detail (good for SR, deblur).

- [ ] **Step 1: Test**

```python
import torch

from refine.losses.freq import FreqL1Loss
from refine.losses.registry import LossContext


def test_freq_zero_when_equal():
    rgb = torch.rand(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=rgb, clean_rgb=rgb.clone(), degraded_rgb=z,
                      task_ids=torch.tensor([0]), task_names=["sr"])
    assert FreqL1Loss()(ctx).item() < 1e-5


def test_freq_grad():
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    clean = torch.rand(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=pred, clean_rgb=clean, degraded_rgb=z,
                      task_ids=torch.tensor([0]), task_names=["sr"])
    FreqL1Loss()(ctx).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
```

- [ ] **Step 2: Implement `src/refine/losses/freq.py`**

```python
"""L1 between FFT log-magnitude spectra of pred and clean."""
from __future__ import annotations

import torch

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("freq_l1")
class FreqL1Loss(RestorationLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        with torch.amp.autocast(ctx.pred_rgb.device.type, enabled=False):
            pred = torch.fft.rfft2(ctx.pred_rgb.float(), norm="ortho")
            clean = torch.fft.rfft2(ctx.clean_rgb.float(), norm="ortho")
            pred_mag = torch.log1p(pred.abs() + self.eps)
            clean_mag = torch.log1p(clean.abs() + self.eps)
            return (pred_mag - clean_mag).abs().mean()
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_freq_loss.py -q
git add src/refine/losses/freq.py tests/test_freq_loss.py
git commit -m "refine: frequency-domain L1 loss (FFT log-magnitude)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: PSNR / SSIM metrics (no-grad logging)

**Files:**
- Create: `src/refine/losses/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Test**

```python
import torch

from refine.losses.metrics import psnr, ssim, per_task_average


def test_psnr_identical_is_inf():
    rgb = torch.rand(1, 3, 16, 16)
    out = psnr(rgb, rgb)
    assert out.item() > 60.0  # epsilon-bounded "infinite"


def test_psnr_decreases_with_noise():
    rgb = torch.rand(1, 3, 16, 16)
    noisy = (rgb + torch.randn_like(rgb) * 0.1).clamp(0, 1)
    assert psnr(noisy, rgb).item() < psnr(rgb, rgb).item()


def test_ssim_shape():
    a = torch.rand(2, 3, 32, 32); b = torch.rand(2, 3, 32, 32)
    s = ssim(a, b)
    assert s.shape == (2,)
    assert (s >= -1.0).all() and (s <= 1.0).all()


def test_per_task_average():
    values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    task_ids = torch.tensor([0, 1, 0, 1])
    out = per_task_average(values, task_ids, num_tasks=2)
    assert out.tolist() == [20.0, 30.0]  # mean by task
```

- [ ] **Step 2: Implement `src/refine/losses/metrics.py`**

```python
"""Per-sample PSNR / SSIM (no grad) + per-task averaging helper."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def psnr(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-batch (B,) PSNR in dB."""
    mse = (pred.float() - clean.float()).pow(2).flatten(1).mean(dim=1)
    eps = 1e-10
    return 10.0 * torch.log10(max_val**2 / (mse + eps))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)  # (size, size)


@torch.no_grad()
def ssim(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-batch (B,) SSIM, averaged over channels."""
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    kernel = _gaussian_kernel().to(pred.device, pred.dtype)
    kernel = kernel.expand(pred.shape[1], 1, *kernel.shape).contiguous()
    pad = kernel.shape[-1] // 2
    g = pred.groups if hasattr(pred, "groups") else pred.shape[1]

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])

    mu_x = conv(pred); mu_y = conv(clean)
    mu_x2 = mu_x.pow(2); mu_y2 = mu_y.pow(2); mu_xy = mu_x * mu_y
    sigma_x2 = conv(pred * pred) - mu_x2
    sigma_y2 = conv(clean * clean) - mu_y2
    sigma_xy = conv(pred * clean) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / den).mean(dim=(1, 2, 3))


def per_task_average(values: torch.Tensor, task_ids: torch.Tensor, num_tasks: int) -> torch.Tensor:
    """values (B,), task_ids (B,) → (num_tasks,) average. NaN where no samples."""
    out = torch.full((num_tasks,), float("nan"), dtype=torch.float32)
    for t in range(num_tasks):
        mask = task_ids == t
        if mask.any():
            out[t] = values[mask].float().mean().item()
    return out
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_metrics.py -q
git add src/refine/losses/metrics.py tests/test_metrics.py
git commit -m "refine: PSNR/SSIM metrics + per-task averaging helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: LossSet composer with `apply_to_tasks` masking

**Files:**
- Create: `src/refine/losses/__init__.py`
- Create: `tests/test_loss_set.py`

- [ ] **Step 1: Test**

```python
import torch

from refine.config import LossConfig
from refine.losses import LossSet
from refine.losses.registry import LossContext


def _ctx_two_tasks():
    rgb = torch.ones(2, 3, 4, 4)
    z = torch.zeros(2, 3, 4, 4)
    return LossContext(pred_rgb=rgb, clean_rgb=z, degraded_rgb=z,
                       task_ids=torch.tensor([0, 1]), task_names=["colorize", "denoise"])


def test_l1_aggregates_with_weight():
    ls = LossSet([LossConfig(name="l1_rgb", weight=2.0)])
    total, log = ls(_ctx_two_tasks())
    assert total.item() == 2.0
    assert log["l1_rgb"] == 1.0


def test_apply_to_tasks_filters_samples():
    """colorfulness restricted to colorize task only — only sample 0 contributes."""
    ls = LossSet([LossConfig(name="colorfulness", weight=1.0, apply_to_tasks=["colorize"])])
    total, log = ls(_ctx_two_tasks())
    assert "colorfulness" in log
    # Sample 0 (colorize) is fully bright (1,1,1), no chroma — cf metric is 0
    # Sample 1 (denoise) is skipped → mean of [0] = 0
    # We just care the loss doesn't blow up and is finite
    assert torch.isfinite(total)


def test_apply_to_tasks_empty_mask():
    """Loss restricted to a task that's not present in the batch returns 0."""
    ls = LossSet([LossConfig(name="l1_rgb", weight=1.0, apply_to_tasks=["jpeg"])])
    total, log = ls(_ctx_two_tasks())
    assert total.item() == 0.0
    assert log["l1_rgb"] == 0.0


def test_has_gan_detected():
    ls_yes = LossSet([LossConfig(name="gan", weight=1.0, config={"gan_type": "hinge"})])
    ls_no = LossSet([LossConfig(name="l1_rgb", weight=1.0)])
    assert ls_yes.has_gan and not ls_no.has_gan
```

- [ ] **Step 2: Implement `src/refine/losses/__init__.py`**

```python
"""Loss aggregation. Imports every loss module so registry is populated."""
from __future__ import annotations

import torch

from refine.config import LossConfig

from . import colorfulness as _colorfulness  # noqa: F401
from . import freq as _freq  # noqa: F401
from . import gan as _gan  # noqa: F401
from . import perceptual as _perceptual  # noqa: F401
from . import pixel as _pixel  # noqa: F401
from .gan import GeneratorGANLoss
from .registry import LossContext, build_loss


class LossSet:
    """Composes weighted losses with optional per-task masks."""

    def __init__(self, configs: list[LossConfig]):
        self.entries: list[tuple[float, object, list[str] | None]] = []
        self.has_gan = False
        self.discriminator_cfg: dict | None = None
        for c in configs:
            loss = build_loss(c.name, c.config)
            self.entries.append((float(c.weight), loss, c.apply_to_tasks))
            if isinstance(loss, GeneratorGANLoss):
                self.has_gan = True
                self.discriminator_cfg = loss.disc_config

    def parameters(self):
        for _, loss, _ in self.entries:
            for p in loss.parameters():
                yield p

    def to(self, device, dtype=None):
        for _, loss, _ in self.entries:
            loss.to(device, dtype) if dtype is not None else loss.to(device)
        return self

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        total: torch.Tensor | float = 0.0
        log: dict[str, float] = {}
        for weight, loss, mask in self.entries:
            if mask is None:
                # all samples
                val = loss(ctx)
            else:
                # select rows whose task name is in mask
                idxs = [i for i, n in enumerate(ctx.task_names) if n in mask]
                if len(idxs) == 0:
                    log[loss.name] = 0.0
                    continue
                idx_t = torch.tensor(idxs, device=ctx.pred_rgb.device)
                sub_ctx = LossContext(
                    pred_rgb=ctx.pred_rgb.index_select(0, idx_t),
                    clean_rgb=ctx.clean_rgb.index_select(0, idx_t),
                    degraded_rgb=ctx.degraded_rgb.index_select(0, idx_t),
                    task_ids=ctx.task_ids.index_select(0, idx_t),
                    task_names=[ctx.task_names[i] for i in idxs],
                    discriminator=ctx.discriminator,
                )
                val = loss(sub_ctx)
            total = total + weight * val
            log[loss.name] = float(val.detach())
        if isinstance(total, float):
            total = torch.zeros((), device=ctx.pred_rgb.device)
        return total, log


__all__ = ["LossSet", "LossContext", "build_loss"]
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_loss_set.py -q
git add src/refine/losses/__init__.py tests/test_loss_set.py
git commit -m "refine: LossSet with apply_to_tasks per-loss masking

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: EMA + checkpoint + task-map sidecar

**Files:**
- Create: `src/refine/train/ema.py`
- Create: `src/refine/train/checkpoint.py`
- Create: `tests/test_train_ckpt_ema.py`

Both lifted from v1. Checkpoint extended to write a `<ckpt>.task_map.json` sidecar.

- [ ] **Step 1: Test**

```python
import json

import torch
from torch import nn

from refine.train.checkpoint import load_checkpoint, save_checkpoint
from refine.train.ema import ModelEMA


def test_ema_converges():
    m = nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0); m.bias.fill_(0.0)
    for _ in range(20):
        ema.update(m)
    assert torch.allclose(ema.module.weight, m.weight, atol=1e-3)


def test_ckpt_with_task_map(tmp_path):
    m = nn.Linear(4, 2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=m, step=10, extra={"foo": "bar"},
                    task_map={"tasks": {"colorize": 0, "sr_x4": 1}})
    sidecar = path.with_suffix(".task_map.json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["tasks"]["colorize"] == 0

    m2 = nn.Linear(4, 2)
    payload = load_checkpoint(path, model=m2)
    assert payload["step"] == 10
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p.data, q.data)
```

- [ ] **Step 2: Implement `src/refine/train/ema.py`**

```python
"""Fp32 ModelEMA shadow."""
from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).float()
        self.module.train(False)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        esd = self.module.state_dict()
        for k, ev in esd.items():
            mv = msd[k].detach()
            if ev.dtype.is_floating_point:
                ev.mul_(self.decay).add_(mv.to(ev.dtype), alpha=1.0 - self.decay)
            else:
                ev.copy_(mv)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
```

- [ ] **Step 3: Implement `src/refine/train/checkpoint.py`**

```python
"""Atomic checkpoint save/load + task-map JSON sidecar."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path, *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    ema=None,
    scheduler=None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
    task_map: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict(), "step": step, "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if optimizer_d is not None:
        payload["optimizer_d"] = optimizer_d.state_dict()
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if task_map is not None:
        payload["task_map"] = task_map

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

    # Sidecar JSON
    if task_map is not None:
        sidecar = path.with_suffix(".task_map.json")
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(task_map, indent=2))
        os.replace(sidecar_tmp, sidecar)


def load_checkpoint(
    path: str | Path, *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    ema=None,
    scheduler=None,
    map_location="cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if optimizer_d is not None and "optimizer_d" in payload:
        optimizer_d.load_state_dict(payload["optimizer_d"])
    if discriminator is not None and "discriminator" in payload:
        discriminator.load_state_dict(payload["discriminator"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_train_ckpt_ema.py -q
git add src/refine/train/ema.py src/refine/train/checkpoint.py tests/test_train_ckpt_ema.py
git commit -m "refine: EMA + atomic checkpoint with task-map JSON sidecar

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: Preview writer (multi-task grid)

**Files:**
- Create: `src/refine/train/preview.py`
- Create: `tests/test_preview.py`

One row per task. Each row: clean | degraded | predicted | |Δ| heatmap.

- [ ] **Step 1: Test**

```python
import numpy as np
import torch

from refine.train.preview import render_multitask_grid, write_png_atomic


def test_render_grid_shape():
    samples = {
        "colorize": [{"clean": torch.rand(3, 32, 32), "degraded": torch.rand(3, 32, 32),
                      "predicted": torch.rand(3, 32, 32)}],
        "denoise":  [{"clean": torch.rand(3, 32, 32), "degraded": torch.rand(3, 32, 32),
                      "predicted": torch.rand(3, 32, 32)}],
    }
    img = render_multitask_grid(samples, caption="step 100", cell_size=32)
    assert img.dtype == np.uint8
    assert img.shape[1] == 32 * 4
    # caption strip + 2 rows of 32px
    assert img.shape[0] >= 32 * 2


def test_atomic_write(tmp_path):
    img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    write_png_atomic(p, img)
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()
```

- [ ] **Step 2: Implement `src/refine/train/preview.py`**

```python
"""Multi-task preview grid renderer + atomic PNG writer."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch


def _t_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).round().astype(np.uint8)


def _delta_heatmap(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 0.5) -> np.ndarray:
    delta = (pred - clean).detach().float().cpu()
    mag = torch.linalg.vector_norm(delta, dim=0)
    mag = (mag.clamp(0, max_val) / max_val * 255.0).to(torch.uint8).numpy()
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def render_multitask_grid(
    samples: dict[str, list[dict[str, torch.Tensor]]],
    *,
    caption: str,
    cell_size: int = 256,
) -> np.ndarray:
    """samples: {task_name: [{"clean":..., "degraded":..., "predicted":...}, ...]}.

    Each task row contains all samples (concatenated horizontally) of:
    clean | degraded | predicted | |Δ| heatmap.
    Tasks stacked vertically. Top caption strip.
    """
    rows: list[np.ndarray] = []
    for task_name, sample_list in samples.items():
        per_sample_rows: list[np.ndarray] = []
        for s in sample_list:
            tiles = []
            for key in ("clean", "degraded", "predicted"):
                img = _t_to_uint8_rgb(s[key])
                if img.shape[:2] != (cell_size, cell_size):
                    img = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
                tiles.append(img)
            delta = _delta_heatmap(s["predicted"], s["clean"])
            if delta.shape[:2] != (cell_size, cell_size):
                delta = cv2.resize(delta, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            per_sample_rows.append(np.concatenate(tiles + [delta], axis=1))
        row = np.concatenate(per_sample_rows, axis=0)
        # Label strip on left (24 px wide)
        label = np.zeros((row.shape[0], 28, 3), dtype=np.uint8)
        cv2.putText(label, task_name[:10], (2, row.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([label, row], axis=1))

    body = np.concatenate(rows, axis=0)
    cap_h = 24
    cap = np.zeros((cap_h, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(cap, caption, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([cap, body], axis=0)


def write_png_atomic(path: str | Path, img_rgb_uint8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_preview.py -q
git add src/refine/train/preview.py tests/test_preview.py
git commit -m "refine: multi-task preview grid renderer (1 row per task)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 22: Rich live UI

**Files:**
- Create: `src/refine/train/ui.py`
- Create: `tests/test_ui_smoke.py`

Lifted from v1, extended with per-task PSNR panel.

- [ ] **Step 1: Test**

```python
from refine.train.ui import TrainUI


def test_ui_renders_with_per_task_metrics():
    ui = TrainUI(run_name="t", total_steps=100, headless=True, task_names=["colorize", "denoise"])
    ui.tick(step=1, losses={"l1_rgb": 0.3, "perceptual_vgg16bn": 0.1}, lr=1e-4,
            throughput_imgs=100.0, per_task_psnr={"colorize": 20.5, "denoise": 28.1})
    frame = ui.render()
    assert frame is not None
```

- [ ] **Step 2: Implement `src/refine/train/ui.py`**

```python
"""Rich live dashboard with per-task PSNR rows."""
from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from refine.utils.gpu import gpu_stats
from refine.utils.timing import EMA


@dataclass
class _EMATrack:
    short: EMA = field(default_factory=lambda: EMA(alpha=0.1))
    long: EMA = field(default_factory=lambda: EMA(alpha=0.01))


class TrainUI(AbstractContextManager):
    def __init__(self, *, run_name: str, total_steps: int, headless: bool = False,
                 task_names: list[str] | None = None) -> None:
        self.run_name = run_name
        self.total_steps = total_steps
        self.headless = headless
        self.task_names = task_names or []
        self.console = Console()
        self._losses: dict[str, _EMATrack] = {}
        self._psnr: dict[str, _EMATrack] = {n: _EMATrack() for n in self.task_names}
        self._lr = 0.0
        self._throughput = 0.0
        self._step = 0
        self._last_preview = ""
        self._t0 = time.perf_counter()
        self._live: Live | None = None
        self._progress = Progress(
            TextColumn("step {task.completed}/{task.total}"),
            BarColumn(),
            TextColumn("{task.percentage:>5.1f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self._task_id = self._progress.add_task("train", total=total_steps)

    def __enter__(self) -> "TrainUI":
        if not self.headless:
            self._live = Live(self.render(), refresh_per_second=6, console=self.console)
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live:
            self._live.__exit__(*exc)
            self._live = None

    def tick(self, *, step: int, losses: dict[str, float], lr: float,
             throughput_imgs: float, per_task_psnr: dict[str, float] | None = None) -> None:
        self._step = step
        self._lr = lr
        self._throughput = throughput_imgs
        for k, v in losses.items():
            t = self._losses.setdefault(k, _EMATrack())
            t.short.update(v); t.long.update(v)
        if per_task_psnr:
            for k, v in per_task_psnr.items():
                if v == v:  # skip NaN
                    track = self._psnr.setdefault(k, _EMATrack())
                    track.short.update(v); track.long.update(v)
        self._progress.update(self._task_id, completed=step)
        if self._live:
            self._live.update(self.render())

    def note_preview(self, msg: str) -> None:
        self._last_preview = msg
        if self._live:
            self._live.update(self.render())

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(Panel.fit(f"run: {self.run_name}", title="refine train"), size=3),
            Layout(self._progress, size=3),
            Layout(name="middle", size=20),
            Layout(Panel.fit(self._last_preview or "(no preview yet)", title="last preview"), size=3),
        )
        layout["middle"].split_row(self._losses_panel(), self._psnr_panel(), self._gpu_panel())
        return layout

    def _losses_panel(self) -> Panel:
        t = Table.grid(padding=(0, 1))
        t.add_column("loss"); t.add_column("value", justify="right"); t.add_column("trend", justify="right")
        for name, tr in self._losses.items():
            s = tr.short.value or 0.0
            l = tr.long.value or s
            arrow = "▼" if s < l else "▲"
            t.add_row(name, f"{s:.4f}", f"{arrow} {abs(s - l):.4f}")
        t.add_row("lr", f"{self._lr:.2e}", "")
        t.add_row("img/s", f"{self._throughput:.1f}", "")
        return Panel(t, title="losses (EMA)")

    def _psnr_panel(self) -> Panel:
        t = Table.grid(padding=(0, 1))
        t.add_column("task"); t.add_column("PSNR", justify="right"); t.add_column("trend", justify="right")
        for name, tr in self._psnr.items():
            s = tr.short.value
            l = tr.long.value
            if s is None:
                t.add_row(name, "—", "")
                continue
            arrow = "▲" if s > (l or s) else "▼"
            t.add_row(name, f"{s:.1f} dB", f"{arrow} {abs(s - (l or s)):.2f}")
        return Panel(t, title="per-task PSNR")

    def _gpu_panel(self) -> Panel:
        s = gpu_stats(0)
        if s is None:
            return Panel("gpu stats unavailable", title="gpu")
        t = Table.grid(padding=(0, 1))
        t.add_column(); t.add_column(justify="right")
        t.add_row("name", s.name[:24])
        t.add_row("mem", f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GB")
        t.add_row("util", f"{s.util_pct}%")
        t.add_row("temp", f"{s.temp_c}°C")
        t.add_row("pwr", f"{s.power_w:.0f}/{s.power_limit_w:.0f} W")
        return Panel(t, title="gpu")
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_ui_smoke.py -q
git add src/refine/train/ui.py tests/test_ui_smoke.py
git commit -m "refine: Rich live UI with per-task PSNR panel

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 23: Trainer (the main loop)

**Files:**
- Create: `src/refine/train/trainer.py`
- Modify: `src/refine/train/__init__.py`
- Create: `tests/test_train_step.py`

Lifted from v1 with all the bf16 stability lessons (NaN-loss guard, grad-norm guard, 20-consecutive-bad abort). Diff vs v1: dataset returns `clean/degraded/task_id/task_name`; model takes `(rgb, task_id)`; loss context is the new dataclass; per-task PSNR aggregation; multi-task preview build.

- [ ] **Step 1: Test (overfit on single batch with 2-task mix)**

```python
import torch

from refine.config import (
    Config, DataConfig, DegradationConfig, LoaderConfig, LossConfig, ModelConfig,
    OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
)
from refine.train.trainer import Trainer


def _make_cfg(image_dir, out_dir):
    return Config(
        run=RunConfig(name="t", output_dir=str(out_dir), seed=0),
        model=ModelConfig(type="nafnet", size="tiny", nf=8,
                          enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                          task_embed_dim=16),
        data=DataConfig(
            root=str(image_dir),
            val_fraction=0.0,
            num_fixed_preview_samples=0,
            num_random_preview_samples=0,
            loader=LoaderConfig(batch_size=2, num_workers=0, persistent_workers=False),
        ),
        degradations={
            "colorize": DegradationConfig(weight=1.0),
            "denoise":  DegradationConfig(weight=1.0, sigma_range=[0.02, 0.05]),
        },
        losses=[LossConfig(name="l1_rgb", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=10),
        train=TrainConfig(total_steps=10, amp="fp32", memory_format="contiguous",
                          compile=False, ema_decay=0.0, preview_every_s=0,
                          ckpt_every_steps=10000, log_every_steps=1),
    )


def test_trainer_overfit_reduces_loss(tmp_image_dir, tmp_path):
    cfg = _make_cfg(tmp_image_dir, tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    batch = next(trainer._iter)
    initial = trainer._train_step(batch)
    for _ in range(30):
        last = trainer._train_step(batch)
    assert last["total_g"] < initial["total_g"]
```

- [ ] **Step 2: Implement `src/refine/train/trainer.py`**

```python
"""Single-class trainer wiring data + model + losses + UI + preview."""
from __future__ import annotations

import math
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from refine.config import Config
from refine.data.dataset import RecursiveImageDataset
from refine.data.degradations.registry import build_degradation
from refine.data.multitask import MultiTaskWrapper, collate_multitask
from refine.losses import LossContext, LossSet
from refine.losses.gan import discriminator_loss
from refine.losses.metrics import per_task_average, psnr
from refine.models import build_model
from refine.models.discriminator import UNetDiscriminator

from .checkpoint import save_checkpoint
from .ema import ModelEMA
from .preview import render_multitask_grid, write_png_atomic
from .ui import TrainUI


def _build_optimizer(params, cfg) -> torch.optim.Optimizer:
    klass = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}[cfg.type]
    kw: dict = {"lr": cfg.lr, "weight_decay": cfg.weight_decay, "betas": tuple(cfg.betas)}
    try:
        kw["fused"] = cfg.fused
    except TypeError:
        pass
    try:
        return klass(params, **kw)
    except (TypeError, ValueError):
        kw.pop("fused", None)
        return klass(params, **kw)


def _build_scheduler(opt, cfg, total_steps):
    if cfg.type == "cosine":
        warm = max(1, cfg.warmup_steps)
        def lr(step):
            if step < warm:
                return step / warm
            t = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr)
    if cfg.type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.milestones), gamma=cfg.gamma)
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, cfg: Config, *, device: torch.device | None = None,
                 headless: bool = False) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.run.seed)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Build degradations + assign task IDs in YAML declaration order
        deg_items = list(cfg.degradations.items())
        self.degradations = []
        self.task_name_to_id: dict[str, int] = {}
        for i, (name, dcfg) in enumerate(deg_items):
            kwargs = {k: v for k, v in dcfg.model_dump(exclude_none=True).items() if k != "weight"}
            deg = build_degradation(name, kwargs)
            deg.task_id = i
            self.degradations.append(deg)
            self.task_name_to_id[name] = i
        weights = [cfg.degradations[name].weight for name, _ in deg_items]
        num_tasks = len(self.degradations)

        # Model
        self.memory_format = (torch.channels_last if cfg.train.memory_format == "channels_last"
                              else torch.contiguous_format)
        self.model = build_model(cfg.model, num_tasks=num_tasks).to(
            self.device, memory_format=self.memory_format)

        # Optim + scheduler
        self.opt_g = _build_optimizer(self.model.parameters(), cfg.optim_g)
        self.scheduler_g = _build_scheduler(self.opt_g, cfg.scheduler, cfg.train.total_steps)

        # Losses
        self.loss_set = LossSet(cfg.losses)
        for _, loss, _ in self.loss_set.entries:
            loss.to(self.device)

        # Discriminator
        self.disc: nn.Module | None = None
        self.opt_d: torch.optim.Optimizer | None = None
        self.gan_type = "hinge"
        if self.loss_set.has_gan:
            dcfg = self.loss_set.discriminator_cfg or {}
            self.disc = UNetDiscriminator(in_ch=3, nf=int(dcfg.get("nf", 64))).to(self.device)
            self.opt_d = _build_optimizer(self.disc.parameters(), cfg.optim_d)
            for _, loss, _ in self.loss_set.entries:
                if hasattr(loss, "gan_type"):
                    self.gan_type = loss.gan_type
                    break

        # EMA
        self.ema = (ModelEMA(self.model, decay=cfg.train.ema_decay)
                    if cfg.train.ema_decay > 0 else None)

        # AMP
        amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
        self.amp_dtype = amp_map[cfg.train.amp]
        self.scaler = (torch.amp.GradScaler("cuda")
                       if (cfg.train.amp == "fp16" and self.device.type == "cuda") else None)

        # Data
        clean = RecursiveImageDataset(
            cfg.data.root, target_size=cfg.model.input_size,
            val_fraction=cfg.data.val_fraction,
            split="train" if cfg.data.val_fraction > 0 else "all",
            augment_hflip=cfg.data.augment.hflip,
            augment_rotate90=cfg.data.augment.rotate90,
            seed=cfg.run.seed,
        )
        if len(clean) == 0:
            raise RuntimeError(f"no images under {cfg.data.root}")
        train_ds = MultiTaskWrapper(clean, self.degradations, weights, seed=cfg.run.seed)
        bs = cfg.data.loader.batch_size if cfg.data.loader.batch_size != "auto" else 16
        self.train_loader = DataLoader(
            train_ds, batch_size=int(bs), shuffle=True,
            num_workers=cfg.data.loader.num_workers,
            pin_memory=cfg.data.loader.pin_memory and self.device.type == "cuda",
            persistent_workers=cfg.data.loader.persistent_workers and cfg.data.loader.num_workers > 0,
            prefetch_factor=cfg.data.loader.prefetch_factor if cfg.data.loader.num_workers > 0 else None,
            collate_fn=collate_multitask, drop_last=True,
        )

        # Val for preview
        if cfg.data.val_fraction > 0:
            clean_val = RecursiveImageDataset(
                cfg.data.root, target_size=cfg.model.input_size,
                val_fraction=cfg.data.val_fraction, split="val",
                augment_hflip=False, seed=cfg.run.seed,
            )
            if len(clean_val) == 0:
                clean_val = clean
            self.val_ds = MultiTaskWrapper(clean_val, self.degradations, weights, seed=cfg.run.seed)
        else:
            self.val_ds = train_ds

        if cfg.train.compile and self.device.type == "cuda":
            self.model = torch.compile(self.model, mode=cfg.train.compile_mode)

        self.ui = TrainUI(run_name=cfg.run.name or "run",
                          total_steps=cfg.train.total_steps, headless=headless,
                          task_names=list(self.task_name_to_id.keys()))
        self._iter = _cycle(self.train_loader)
        self.step = 0
        self._last_preview_t = 0.0
        self._t_window = time.perf_counter()
        self._samples_window = 0
        self._preview_lock = threading.Lock()
        self._consecutive_nan = 0

    def _amp_ctx(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype)

    def run_one_step(self) -> dict[str, float]:
        return self._train_step(next(self._iter))

    def _train_step(self, batch: dict) -> dict[str, float]:
        clean = batch["clean"].to(self.device, non_blocking=True, memory_format=self.memory_format)
        degraded = batch["degraded"].to(self.device, non_blocking=True, memory_format=self.memory_format)
        task_id = batch["task_id"].to(self.device, non_blocking=True)
        task_names = batch["task_name"]

        self.opt_g.zero_grad(set_to_none=True)
        with self._amp_ctx():
            pred = self.model(degraded, task_id)
            ctx = LossContext(pred_rgb=pred, clean_rgb=clean, degraded_rgb=degraded,
                              task_ids=task_id, task_names=task_names, discriminator=self.disc)
            total_g, log_g = self.loss_set(ctx)
            if not torch.isfinite(total_g):
                self.opt_g.zero_grad(set_to_none=True)
                self.scheduler_g.step()
                self.step += 1
                self._samples_window += degraded.shape[0]
                self._consecutive_nan += 1
                if self._consecutive_nan >= 20:
                    raise RuntimeError(
                        f"20 consecutive non-finite losses at step {self.step}. "
                        "Resume from last.pt and try amp=fp32 or lower lr.")
                return {"total_g": float(total_g.detach()), **log_g, "_skipped": 1.0}

        step_skipped = False
        if self.scaler is not None:
            self.scaler.scale(total_g).backward()
            self.scaler.unscale_(self.opt_g)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.cfg.train.clip_grad_norm)
            if torch.isfinite(grad_norm):
                self.scaler.step(self.opt_g)
            else:
                step_skipped = True; self.opt_g.zero_grad(set_to_none=True)
            self.scaler.update()
        else:
            total_g.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.cfg.train.clip_grad_norm)
            if torch.isfinite(grad_norm):
                self.opt_g.step()
            else:
                step_skipped = True; self.opt_g.zero_grad(set_to_none=True)

        if step_skipped:
            self._consecutive_nan += 1
        else:
            self._consecutive_nan = 0
        if self._consecutive_nan >= 20:
            raise RuntimeError(
                f"20 consecutive non-finite gradient steps at step {self.step}. "
                "Resume from last.pt and try amp=fp32 or lower lr.")

        log: dict[str, float] = {"total_g": float(total_g.detach()), **log_g,
                                 "grad_norm": float(grad_norm)}
        if step_skipped:
            log["_skipped_grad"] = 1.0

        # Per-task PSNR (no-grad)
        with torch.no_grad():
            per_sample = psnr(pred, clean)
            num_tasks = len(self.task_name_to_id)
            avg = per_task_average(per_sample.cpu(), task_id.cpu(), num_tasks=num_tasks)
        self._last_per_task_psnr = {
            name: float(avg[i]) for name, i in self.task_name_to_id.items()
        }

        if self.disc is not None and self.opt_d is not None:
            self.opt_d.zero_grad(set_to_none=True)
            with self._amp_ctx():
                d_loss = discriminator_loss(self.disc, clean.detach(), pred.detach(), self.gan_type)
            if self.scaler is not None:
                self.scaler.scale(d_loss).backward(); self.scaler.unscale_(self.opt_d)
                self.scaler.step(self.opt_d); self.scaler.update()
            else:
                d_loss.backward(); self.opt_d.step()
            log["d_total"] = float(d_loss.detach())

        if self.ema is not None:
            self.ema.update(self.model)
        self.scheduler_g.step()
        self.step += 1
        self._samples_window += degraded.shape[0]
        return log

    def fit(self) -> None:
        with self.ui:
            self._last_preview_t = time.perf_counter() - self.cfg.train.preview_every_s
            for _ in range(self.cfg.train.total_steps):
                log = self.run_one_step()
                if self.step % self.cfg.train.log_every_steps == 0:
                    now = time.perf_counter()
                    imgs_per_s = self._samples_window / max(1e-6, now - self._t_window)
                    self._t_window = now; self._samples_window = 0
                    self.ui.tick(step=self.step, losses=log,
                                 lr=self.opt_g.param_groups[0]["lr"],
                                 throughput_imgs=imgs_per_s,
                                 per_task_psnr=getattr(self, "_last_per_task_psnr", None))
                if (self.cfg.train.preview_every_s > 0
                        and time.perf_counter() - self._last_preview_t
                        >= self.cfg.train.preview_every_s):
                    self._write_preview()
                if (self.cfg.train.ckpt_every_steps > 0
                        and self.step % self.cfg.train.ckpt_every_steps == 0):
                    self._save_ckpt(name="last.pt")
            self._save_ckpt(name="final.pt")
            self._write_preview()
            if self.cfg.export.on_finish:
                self._maybe_export_onnx()

    @torch.inference_mode()
    def _build_preview_samples(self) -> dict[str, list[dict]]:
        n_fixed = self.cfg.data.num_fixed_preview_samples
        n_rand = self.cfg.data.num_random_preview_samples
        eval_model = self.ema.module if self.ema is not None else self.model
        was_training = eval_model.training
        eval_model.train(False)

        out: dict[str, list[dict]] = {name: [] for name in self.task_name_to_id}
        deg_by_name = {d.name: d for d in self.degradations}
        n_total = len(self.val_ds.clean)
        idxs = list(range(min(n_fixed, n_total)))
        if n_rand > 0 and n_total > len(idxs):
            extra = torch.randint(len(idxs), n_total, (min(n_rand, n_total - len(idxs)),)).tolist()
            idxs += extra
        import random as _random
        for task_name, deg in deg_by_name.items():
            for i in idxs:
                clean_t = self.val_ds.clean[i]
                rng = _random.Random((self.cfg.run.seed * 1_000_003) ^ (i + deg.task_id))
                degraded_np = deg.degrade(clean_t.permute(1, 2, 0).numpy(), rng)
                degraded_t = torch.from_numpy(degraded_np.transpose(2, 0, 1)).contiguous()
                pred = eval_model(degraded_t.unsqueeze(0).to(self.device),
                                  torch.tensor([deg.task_id], dtype=torch.long, device=self.device))
                out[task_name].append({
                    "clean": clean_t, "degraded": degraded_t,
                    "predicted": pred.clamp(0, 1).squeeze(0).cpu(),
                })
        if was_training:
            eval_model.train(True)
        return out

    def _write_preview(self) -> None:
        with self._preview_lock:
            try:
                samples = self._build_preview_samples()
            except Exception as e:
                self.ui.note_preview(f"preview error: {e}")
                self._last_preview_t = time.perf_counter()
                return
            caption = f"step {self.step}  ts {time.strftime('%H:%M:%S')}"
            grid = render_multitask_grid(samples, caption=caption,
                                          cell_size=self.cfg.model.input_size)
            latest = self.output_dir / "samples" / "latest.png"
            write_png_atomic(latest, grid)
            if (self.cfg.train.preview_history_every > 0
                    and (self.step % self.cfg.train.preview_history_every == 0)):
                hist = self.output_dir / "samples" / f"iter_{self.step:07d}.png"
                write_png_atomic(hist, grid)
            try:
                rel = latest.relative_to(self.output_dir)
            except ValueError:
                rel = latest
            self.ui.note_preview(f"wrote {rel} @ step {self.step}")
            self._last_preview_t = time.perf_counter()

    def _task_map(self) -> dict:
        return {
            "tasks": dict(self.task_name_to_id),
            "input_size": self.cfg.model.input_size,
            "model_size": self.cfg.model.size,
            "version": "0.1.0",
        }

    def _save_ckpt(self, name: str) -> None:
        save_checkpoint(
            self.output_dir / "ckpt" / name,
            model=self.model, optimizer=self.opt_g, optimizer_d=self.opt_d,
            discriminator=self.disc, ema=self.ema, scheduler=self.scheduler_g,
            step=self.step, extra={"cfg": self.cfg.model_dump()},
            task_map=self._task_map(),
        )

    def _maybe_export_onnx(self) -> None:
        try:
            from refine.export.onnx import export_onnx_from_model
        except Exception:
            return
        export_model = self.ema.module if self.ema is not None else self.model
        export_onnx_from_model(
            export_model, num_tasks=len(self.task_name_to_id),
            input_size=self.cfg.model.input_size,
            export_path=self.output_dir / "model.onnx",
            opset=self.cfg.export.opset, simplify=self.cfg.export.simplify,
            task_map=self._task_map(),
        )
        if self.cfg.export.dynamic_hw:
            export_onnx_from_model(
                export_model, num_tasks=len(self.task_name_to_id),
                input_size=self.cfg.model.input_size,
                export_path=self.output_dir / "model_dynamic.onnx",
                opset=self.cfg.export.opset, simplify=self.cfg.export.simplify,
                dynamic_hw=True, task_map=self._task_map(),
            )


def fit(cfg: Config, *, device: torch.device | None = None) -> None:
    Trainer(cfg, device=device).fit()
```

- [ ] **Step 3: Update `src/refine/train/__init__.py`**

```python
from .trainer import Trainer, fit
__all__ = ["Trainer", "fit"]
```

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/test_train_step.py -q
git add src/refine/train tests/test_train_step.py
git commit -m "refine: Trainer wires data/model/losses/UI/preview for multi-task

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 24: Inference pipeline (rgb + task → rgb, reflection padding)

**Files:**
- Create: `src/refine/infer/pipeline.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Test**

```python
import numpy as np
import torch

from refine.config import ModelConfig
from refine.infer.pipeline import MultiTaskRefinerPipeline, pad_to_multiple, unpad
from refine.models import build_model


def test_pad_unpad_round_trip():
    img = np.random.rand(45, 71, 3).astype(np.float32)
    padded, pads = pad_to_multiple(img, multiple=16, mode="reflect")
    assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0
    back = unpad(padded, *pads)
    np.testing.assert_array_equal(back, img)


def test_pipeline_rgb_to_rgb_shape():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_tasks=2)
    pipe = MultiTaskRefinerPipeline(m, task_name_to_id={"colorize": 0, "denoise": 1},
                                     device=torch.device("cpu"))
    img = (np.random.rand(33, 55, 3) * 255).astype(np.uint8)
    out = pipe.process(img, task="colorize")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Implement `src/refine/infer/pipeline.py`**

```python
"""Inference pipeline: RGB-in, RGB-out, task-aware."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def pad_to_multiple(rgb: np.ndarray, *, multiple: int = 16,
                     mode: str = "reflect") -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    padded = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode)
    return padded, (pad_t, pad_b, pad_l, pad_r)


def unpad(img: np.ndarray, pad_t: int, pad_b: int, pad_l: int, pad_r: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[pad_t : h - pad_b, pad_l : w - pad_r]


class MultiTaskRefinerPipeline:
    def __init__(self, model: nn.Module, *, task_name_to_id: dict[str, int],
                 device: torch.device | None = None) -> None:
        self.task_name_to_id = dict(task_name_to_id)
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.train(False)

    @torch.inference_mode()
    def process(self, img_bgr: np.ndarray, *, task: str) -> np.ndarray:
        if task not in self.task_name_to_id:
            raise ValueError(f"unknown task {task!r}; have {list(self.task_name_to_id)}")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_padded, pads = pad_to_multiple(rgb, multiple=16, mode="reflect")
        t = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        task_id = torch.tensor([self.task_name_to_id[task]], dtype=torch.long, device=self.device)
        out = self.model(t, task_id).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = unpad(out, *pads)
        return (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255.0).round().clip(0, 255).astype(np.uint8)


def load_pipeline(checkpoint: str | Path, *, device: torch.device | None = None) -> MultiTaskRefinerPipeline:
    from refine.config import ModelConfig
    from refine.models import build_model

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg_dict = cfg_dict.get("model") or {}
    mcfg = ModelConfig(**mcfg_dict)
    task_map = payload.get("task_map") or {}
    task_name_to_id = task_map.get("tasks") or {"colorize": 0}
    model = build_model(mcfg, num_tasks=len(task_name_to_id))
    model.load_state_dict(payload["model"])
    return MultiTaskRefinerPipeline(model, task_name_to_id=task_name_to_id, device=device)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_inference.py -q
git add src/refine/infer/pipeline.py tests/test_inference.py
git commit -m "refine: multi-task inference pipeline (rgb + task -> rgb, reflect-pad)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 25: ONNX export (2 inputs, per-task parity)

**Files:**
- Create: `src/refine/export/onnx.py`
- Create: `tests/test_export_onnx.py`

- [ ] **Step 1: Test**

```python
import json
import os

import pytest


@pytest.mark.skipif(os.environ.get("REFINE_SLOW") != "1",
                    reason="onnx export is slow; set REFINE_SLOW=1 to run")
def test_onnx_export_parity_all_tasks(tmp_path):
    import numpy as np
    import torch

    from refine.config import ModelConfig
    from refine.export.onnx import export_onnx_from_model
    from refine.models import build_model

    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    num_tasks = 3
    m = build_model(cfg, num_tasks=num_tasks)
    m.train(False)
    path = tmp_path / "m.onnx"
    task_map = {"tasks": {"colorize": 0, "denoise": 1, "sr_x4": 2}}
    export_onnx_from_model(m, num_tasks=num_tasks, input_size=32, export_path=path,
                            opset=17, simplify=False, task_map=task_map)
    assert path.exists()
    sidecar = path.with_suffix(".task_map.json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["tasks"]["sr_x4"] == 2

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for tid in range(num_tasks):
        x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        t = np.array([tid], dtype=np.int64)
        onnx_out = sess.run(None, {"input": x, "task": t})[0]
        with torch.no_grad():
            torch_out = m(torch.from_numpy(x), torch.from_numpy(t)).numpy()
        np.testing.assert_allclose(onnx_out, torch_out, atol=1e-3, rtol=1e-2)
```

- [ ] **Step 2: Implement `src/refine/export/onnx.py`**

```python
"""Export multi-task refine model to ONNX with per-task parity verification."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def export_onnx_from_model(
    model: nn.Module, *,
    num_tasks: int,
    input_size: int,
    export_path: str | Path,
    opset: int = 17,
    simplify: bool = True,
    verify_parity: bool = True,
    parity_atol: float = 1e-3,
    dynamic_hw: bool = False,
    task_map: dict | None = None,
) -> None:
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.train(False)

    dummy_rgb = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    dummy_task = torch.tensor([0], dtype=torch.long)

    dynamic_axes: dict[str, dict[int, str]] = {
        "input":  {0: "batch"},
        "task":   {0: "batch"},
        "output": {0: "batch"},
    }
    if dynamic_hw:
        dynamic_axes["input"][2] = "height"; dynamic_axes["input"][3] = "width"
        dynamic_axes["output"][2] = "height"; dynamic_axes["output"][3] = "width"

    torch.onnx.export(
        model, (dummy_rgb, dummy_task), str(export_path),
        opset_version=opset,
        input_names=["input", "task"], output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if simplify:
        try:
            import onnx
            import onnxsim
            m_onnx = onnx.load(str(export_path))
            m_onnx, ok = onnxsim.simplify(m_onnx)
            if ok:
                onnx.save(m_onnx, str(export_path))
        except Exception:
            pass

    if verify_parity:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        # Parity at export size, every task
        for tid in range(num_tasks):
            x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
            t = np.array([tid], dtype=np.int64)
            ort_out = sess.run(None, {"input": x, "task": t})[0]
            with torch.no_grad():
                t_out = model(torch.from_numpy(x), torch.from_numpy(t)).numpy()
            diff = float(np.abs(ort_out - t_out).max())
            if diff > parity_atol:
                raise RuntimeError(
                    f"ONNX parity failed for task {tid}: max_abs_diff={diff:.3e}")
        # Non-square dynamic-hw spot-check
        if dynamic_hw:
            alt_h = max(48, input_size // 2); alt_w = max(48, input_size // 2 + 32)
            for tid in range(num_tasks):
                x = np.random.rand(1, 3, alt_h, alt_w).astype(np.float32)
                t = np.array([tid], dtype=np.int64)
                try:
                    ort_out = sess.run(None, {"input": x, "task": t})[0]
                except Exception as e:
                    raise RuntimeError(f"dynamic_hw ONNX rejected {alt_h}x{alt_w}: {e}") from e
                with torch.no_grad():
                    t_out = model(torch.from_numpy(x), torch.from_numpy(t)).numpy()
                diff = float(np.abs(ort_out - t_out).max())
                if diff > parity_atol:
                    raise RuntimeError(
                        f"dynamic-hw parity failed for task {tid} at {alt_h}x{alt_w}: "
                        f"max_abs_diff={diff:.3e}")

    if task_map is not None:
        sidecar = export_path.with_suffix(".task_map.json")
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(task_map, indent=2))
        os.replace(sidecar_tmp, sidecar)
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_export_onnx.py -q
REFINE_SLOW=1 uv run pytest tests/test_export_onnx.py -q
git add src/refine/export/onnx.py tests/test_export_onnx.py
git commit -m "refine: ONNX export with 2 inputs and per-task parity verification

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 26: Default + tiny + large + laion-multitask configs

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/tiny.yaml`
- Create: `configs/large.yaml`
- Create: `configs/laion-multitask.yaml`
- Create: `tests/test_configs_load.py`

- [ ] **Step 1: Test**

```python
from pathlib import Path

from refine.config import load_config

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_tiny_yaml_loads():
    cfg = load_config(ROOT / "tiny.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "tiny"


def test_large_yaml_loads():
    cfg = load_config(ROOT / "large.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "large"


def test_laion_multitask_loads():
    cfg = load_config(ROOT / "laion-multitask.yaml", overrides={"data": {"root": "/tmp"}})
    assert "colorize" in cfg.degradations
    assert "sr_x4" in cfg.degradations
```

- [ ] **Step 2: `configs/default.yaml`**

```yaml
run:
  name: "${date:%Y-%m-%d_%H-%M-%S}"
  output_dir: "runs/${date:%Y-%m-%d_%H-%M-%S}"
  seed: 0

model:
  type: nafnet
  size: tiny
  input_size: 256
  task_embed_dim: 128

data:
  val_fraction: 0.01
  num_fixed_preview_samples: 2
  num_random_preview_samples: 1
  augment:
    hflip: true
    rotate90: false
  loader:
    batch_size: 32
    num_workers: 16
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4

degradations:
  colorize: { weight: 1.0 }
  denoise:  { weight: 1.0, sigma_range: [0.005, 0.05] }
  sr_x2:    { weight: 1.0, factor: 2 }
  sr_x4:    { weight: 1.0, factor: 4 }
  deblur:   { weight: 0.7, sigma_range: [1.0, 3.0], motion_prob: 0.2 }
  jpeg:     { weight: 0.7, quality_range: [20, 70] }

losses: !preset standard

optim_g:
  type: AdamW
  lr: 1.0e-4
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
  warmup_steps: 2000
  total_steps: 100000

train:
  total_steps: 100000
  amp: bf16
  memory_format: channels_last
  compile: false
  compile_mode: default
  ema_decay: 0.999
  grad_accum_steps: 1
  clip_grad_norm: 1.0
  preview_every_s: 10.0
  preview_history_every: 50
  ckpt_every_steps: 5000
  val_every_steps: 5000
  log_every_steps: 25

export:
  on_finish: true
  opset: 17
  simplify: true
  dynamic_hw: true
```

- [ ] **Step 3: `configs/tiny.yaml`**

```yaml
defaults: default.yaml
model:
  size: tiny
data:
  loader:
    batch_size: 32
```

- [ ] **Step 4: `configs/large.yaml`**

```yaml
defaults: default.yaml
model:
  size: large
data:
  loader:
    batch_size: 12
scheduler:
  warmup_steps: 3000
```

- [ ] **Step 5: `configs/laion-multitask.yaml`**

```yaml
defaults: large.yaml
run:
  name: "laion-multitask-${date:%Y-%m-%d_%H-%M-%S}"
data:
  val_fraction: 0.005
  num_fixed_preview_samples: 1
  num_random_preview_samples: 1
  loader:
    batch_size: 12
    num_workers: 16
```

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/test_configs_load.py -q
git add configs tests/test_configs_load.py
git commit -m "refine: default/tiny/large/laion-multitask configs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 27: CLI (train, infer, export, scan-data, list-tasks)

**Files:**
- Modify: `src/refine/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Test**

```python
from pathlib import Path

from typer.testing import CliRunner

from refine.cli import app

runner = CliRunner()


def test_help_top_level():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for sub in ("train", "infer", "export", "scan-data", "list-tasks"):
        assert sub in r.stdout


def test_help_train():
    r = runner.invoke(app, ["train", "--help"])
    assert r.exit_code == 0


def test_scan_data_writes_manifest(tmp_image_dir):
    r = runner.invoke(app, ["scan-data", "--root", str(tmp_image_dir)])
    assert r.exit_code == 0
    assert (tmp_image_dir / ".refine-manifest.txt").exists()


def test_list_tasks(tmp_path):
    ROOT = Path(__file__).resolve().parents[1] / "configs"
    r = runner.invoke(app, ["list-tasks", "--config", str(ROOT / "tiny.yaml"),
                            "--data", str(tmp_path)])
    assert r.exit_code == 0
    assert "colorize" in r.stdout
    assert "sr_x4" in r.stdout
```

- [ ] **Step 2: Implement `src/refine/cli.py`**

```python
"""refine CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="refine — multi-task image restoration", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """refine CLI."""


@app.command()
def version() -> None:
    from refine import __version__
    typer.echo(__version__)


@app.command(name="scan-data")
def scan_data(root: Path = typer.Option(..., "--root", exists=True, file_okay=False)) -> None:
    from refine.data.dataset import build_manifest
    paths = build_manifest(root, force=True)
    typer.echo(f"{len(paths)} images indexed under {root}")


@app.command(name="list-tasks")
def list_tasks(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    data: Optional[Path] = typer.Option(None, "--data"),
) -> None:
    from refine.config import load_config
    overrides = {"data": {"root": str(data)}} if data else None
    cfg = load_config(config, overrides=overrides)
    for i, (name, dcfg) in enumerate(cfg.degradations.items()):
        typer.echo(f"  [{i}] {name:12s} weight={dcfg.weight}")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    data: Optional[Path] = typer.Option(None, "--data"),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    batch_size: Optional[str] = typer.Option(None, "--batch-size"),
    compile_: bool = typer.Option(False, "--compile/--no-compile"),
    amp: Optional[str] = typer.Option(None, "--amp"),
    total_steps: Optional[int] = typer.Option(None, "--total-steps"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
) -> None:
    from refine.config import load_config
    from refine.train import Trainer
    from refine.train.checkpoint import load_checkpoint

    overrides: dict = {}
    if data is not None:
        overrides.setdefault("data", {})["root"] = str(data)
    if run_name is not None:
        overrides.setdefault("run", {})["name"] = run_name
    if batch_size is not None:
        bs: int | str = "auto" if batch_size == "auto" else int(batch_size)
        overrides.setdefault("data", {}).setdefault("loader", {})["batch_size"] = bs
    if amp is not None:
        overrides.setdefault("train", {})["amp"] = amp
    if total_steps is not None:
        overrides.setdefault("train", {})["total_steps"] = total_steps
        overrides.setdefault("scheduler", {})["total_steps"] = total_steps
    if compile_:
        overrides.setdefault("train", {})["compile"] = True

    cfg = load_config(config, overrides=overrides)
    trainer = Trainer(cfg)
    if resume is not None:
        load_checkpoint(resume, model=trainer.model, optimizer=trainer.opt_g,
                        optimizer_d=trainer.opt_d, discriminator=trainer.disc,
                        ema=trainer.ema, scheduler=trainer.scheduler_g)
    trainer.fit()


@app.command()
def infer(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    input_: Path = typer.Option(..., "--input", "--in", exists=True),
    output: Path = typer.Option(..., "--output", "--out"),
    task: str = typer.Option(..., "--task"),
    upsample_to: Optional[str] = typer.Option(
        None, "--upsample-to",
        help="WxH (e.g. 2048x2048) — bicubic upsample input before inference (for SR tasks)"),
) -> None:
    import cv2
    import torch
    from refine.infer.pipeline import load_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(model, device=device)

    def maybe_upsample(img):
        if not upsample_to:
            return img
        w, h = (int(x) for x in upsample_to.lower().split("x"))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    if input_.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(input_))
        if img is None:
            raise typer.BadParameter(f"could not read {input_}")
        cv2.imwrite(str(output), pipe.process(maybe_upsample(img), task=task))
    else:
        output.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        for p in sorted(input_.rglob("*")):
            if p.suffix.lower() not in exts:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            out_path = output / p.relative_to(input_)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), pipe.process(maybe_upsample(img), task=task))
    typer.echo(f"wrote {output}")


@app.command()
def export(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(256, "--input-size"),
    opset: int = typer.Option(17, "--opset"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify"),
    dynamic_hw: bool = typer.Option(False, "--dynamic-hw/--fixed-hw"),
) -> None:
    import torch
    from refine.config import ModelConfig
    from refine.export.onnx import export_onnx_from_model
    from refine.models import build_model

    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg = ModelConfig(**(cfg_dict.get("model") or {}))
    task_map = payload.get("task_map") or {}
    num_tasks = len(task_map.get("tasks") or {"colorize": 0})
    m = build_model(mcfg, num_tasks=num_tasks)
    m.load_state_dict(payload["model"])
    export_onnx_from_model(
        m, num_tasks=num_tasks, input_size=input_size,
        export_path=output, opset=opset, simplify=simplify,
        dynamic_hw=dynamic_hw, task_map=task_map,
    )
    typer.echo(f"wrote {output}")
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/test_cli.py -q
git add src/refine/cli.py tests/test_cli.py
git commit -m "refine: full Typer CLI (train/infer/export/scan-data/list-tasks)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 28: End-to-end smoke test

**Files:**
- Create: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Test**

```python
import os

import cv2
import numpy as np
import pytest
import torch

from refine.config import (
    Config, DataConfig, DegradationConfig, ExportConfig, LoaderConfig, LossConfig,
    ModelConfig, OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
)
from refine.infer.pipeline import load_pipeline
from refine.train import Trainer


@pytest.mark.skipif(os.environ.get("REFINE_SLOW") != "1",
                    reason="e2e smoke is slow; set REFINE_SLOW=1 to run")
def test_train_then_infer_e2e(tmp_path):
    data_dir = tmp_path / "imgs"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        cv2.imwrite(str(data_dir / f"img{i}.png"),
                    rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8))

    out_dir = tmp_path / "run"
    cfg = Config(
        run=RunConfig(name="smoke", output_dir=str(out_dir), seed=0),
        model=ModelConfig(type="nafnet", size="tiny", nf=8,
                          enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                          task_embed_dim=16, input_size=64),
        data=DataConfig(root=str(data_dir), val_fraction=0.25,
                        num_fixed_preview_samples=1, num_random_preview_samples=0,
                        loader=LoaderConfig(batch_size=2, num_workers=0,
                                            persistent_workers=False)),
        degradations={
            "colorize": DegradationConfig(weight=1.0),
            "denoise":  DegradationConfig(weight=1.0, sigma_range=[0.02, 0.05]),
        },
        losses=[LossConfig(name="l1_rgb", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=5),
        train=TrainConfig(total_steps=5, amp="fp32", memory_format="contiguous",
                          compile=False, ema_decay=0.0, preview_every_s=0.001,
                          preview_history_every=0, ckpt_every_steps=5,
                          log_every_steps=1),
        export=ExportConfig(on_finish=False),
    )
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    trainer.fit()

    final_ckpt = out_dir / "ckpt" / "final.pt"
    assert final_ckpt.exists()
    assert (out_dir / "samples" / "latest.png").exists()
    sidecar = final_ckpt.with_suffix(".task_map.json")
    assert sidecar.exists()

    pipe = load_pipeline(final_ckpt, device=torch.device("cpu"))
    img = cv2.imread(str(data_dir / "img0.png"))
    out = pipe.process(img, task="colorize")
    assert out.shape == img.shape and out.dtype == np.uint8
```

- [ ] **Step 2: Run**

```bash
REFINE_SLOW=1 uv run pytest tests/test_e2e_smoke.py -v
```

Expected: 1 passed (covers training loop, multi-task picker, preview write, ckpt+sidecar, inference pipeline end-to-end).

- [ ] **Step 3: Run full suite**

```bash
uv run pytest -q                # all non-slow tests
REFINE_SLOW=1 uv run pytest -q  # full suite including slow
```

- [ ] **Step 4: Manual sanity training run**

```bash
# Use your existing dataset:
uv run refine scan-data --root ~/data/laion-images
uv run refine train --config configs/tiny.yaml --data ~/data/laion-images \
                    --total-steps 50 --run-name smoke
ls runs/smoke/samples/ runs/smoke/ckpt/
# Expect: latest.png with 6 task rows, ckpt/final.pt and .task_map.json
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "refine: end-to-end smoke test (train+ckpt+preview+infer per task)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Spec coverage check

| Spec section | Implemented by |
|---|---|
| 1 Goal | All tasks together |
| 2 Non-goals | Explicit (no DDP/video/blind-detection v1) |
| 3 Strategy (NAFNet + LAB + FiLM/AdaLN) | Tasks 4, 10–13 |
| 4 Project structure + archival | Task 1 |
| 5 Model architecture | Tasks 4, 10, 11, 12, 13 |
| 6 Degradation pipeline | Tasks 6, 7, 8, 9 |
| 7 Loss system + apply_to_tasks | Tasks 14-19 |
| 8 Trainer | Tasks 20, 21, 22, 23 |
| 9 Inference | Task 24 |
| 10 ONNX export + per-task parity + sidecar | Task 25 |
| 11 CLI | Task 27 |
| 12 Configs | Task 26 |
| 13 Dependencies | Task 1 |
| 14 Testing strategy | Every task ships tests; Task 28 is the integration smoke |
| 15 Migration sequence | Task ordering (1 → 28) |
| 16 Open questions | Deferred — none blocking |





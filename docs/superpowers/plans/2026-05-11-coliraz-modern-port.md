# Coliraz Modern Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port DDColor to a clean, modern PyTorch 2.x codebase managed with `uv`, providing a single `coliraz` CLI for training (with live UI + periodic preview), inference, and ONNX export — tuned for Blackwell 96 GB.

**Architecture:** Fresh rewrite using `timm` ConvNeXt encoder, `nn.MultiheadAttention` (SDPA → FlashAttention on Blackwell), modular toggleable losses, single-class trainer with bf16 AMP + channels-last + EMA + `torch.compile` opt-in. Layered Pydantic+YAML config with `!preset` shortcuts. Rich-based live dashboard and a background-thread preview writer producing periodic PNG comparison grids.

**Tech Stack:** Python 3.11, `uv` package manager, PyTorch 2.4+, `timm`, `typer`, `pydantic`, `pyyaml`, `rich`, `onnx`/`onnxruntime`/`onnxsim`, `opencv-python-headless`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-11-coliraz-modern-port-design.md`

---

## File Structure

```
coliraz/
├── reference/ddcolor_original/        # moved from ./DDColor/
├── src/coliraz/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models/{__init__.py, ddcolor.py, encoder.py, pixel_decoder.py,
│   │           color_decoder.py, unet_blocks.py, refine.py, discriminator.py}
│   ├── losses/{__init__.py, registry.py, pixel.py, perceptual.py,
│   │           gan.py, colorfulness.py}
│   ├── data/{__init__.py, dataset.py, grayscale.py, transforms.py}
│   ├── train/{__init__.py, trainer.py, ui.py, preview.py,
│   │           checkpoint.py, ema.py, auto_batch.py}
│   ├── infer/{__init__.py, pipeline.py}
│   ├── export/{__init__.py, onnx.py}
│   └── utils/{__init__.py, color.py, gpu.py, timing.py}
├── configs/{default.yaml, tiny.yaml, large.yaml}
├── tests/{conftest.py, test_color.py, test_config.py, test_grayscale.py,
│         test_dataset.py, test_transforms.py, test_encoder.py,
│         test_unet_blocks.py, test_pixel_decoder.py, test_color_decoder.py,
│         test_ddcolor.py, test_discriminator.py, test_losses.py,
│         test_loss_set.py, test_ema.py, test_checkpoint.py, test_preview.py,
│         test_ui.py, test_train_step.py, test_inference.py,
│         test_export_onnx.py, test_cli.py}
├── pyproject.toml
├── main.py
└── README.md
```

## Conventions

- All tests run on CPU. Anything that would normally use CUDA is wrapped to use CPU fallback for tests.
- All tests must complete in under 30 s total. Slow tests (perceptual, ONNX export) are skipped without `COLIRAZ_SLOW=1`.
- Every commit message is prefixed `coliraz:` and ends with the Co-Authored-By line.
- Each task ends with running `pytest tests/ -q` to ensure no regressions, then a single commit.

---

## Task 1: Scaffold project structure & dependencies

**Files:**
- Move: `DDColor/` → `reference/ddcolor_original/`
- Modify: `pyproject.toml`
- Modify: `main.py`
- Create: `.gitignore`
- Create: `src/coliraz/__init__.py` and stub `__init__.py` in every subpackage
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `README.md` (one-paragraph stub)

- [ ] **Step 1: Move reference code**

```bash
mkdir -p reference
git mv DDColor reference/ddcolor_original
```

Verify: `ls reference/ddcolor_original/` shows `basicsr ddcolor README.md` etc.

- [ ] **Step 2: Write the new `pyproject.toml`**

Overwrite `pyproject.toml`:

```toml
[project]
name = "coliraz"
version = "0.1.0"
description = "Modern PyTorch port of DDColor for image colorization"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.4",
  "torchvision>=0.19",
  "timm>=1.0",
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
  "pynvml>=11.5",
  "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff>=0.6"]

[project.scripts]
coliraz = "coliraz.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/coliraz"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

- [ ] **Step 3: Replace `main.py`**

```python
"""Thin entry point — delegates to the Typer CLI."""
from coliraz.cli import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.uv/
.pytest_cache/
.ruff_cache/
*.egg-info/
dist/
build/
runs/
samples/
.coliraz-manifest.txt
*.onnx
.cache/
```

- [ ] **Step 5: Create package skeletons**

Create empty `__init__.py` in each of: `src/coliraz/`, `src/coliraz/models/`, `src/coliraz/losses/`, `src/coliraz/data/`, `src/coliraz/train/`, `src/coliraz/infer/`, `src/coliraz/export/`, `src/coliraz/utils/`, `tests/`.

`src/coliraz/__init__.py`:

```python
"""coliraz — modern PyTorch port of DDColor."""
__version__ = "0.1.0"
```

- [ ] **Step 6: Stub a minimum-viable `cli.py` so `coliraz` import works**

`src/coliraz/cli.py`:

```python
import typer

app = typer.Typer(help="coliraz — image colorization")


@app.command()
def version():
    from coliraz import __version__
    typer.echo(__version__)
```

- [ ] **Step 7: Write `tests/conftest.py`**

```python
import pathlib
import sys

import numpy as np
import pytest


@pytest.fixture
def tmp_image_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tmp directory containing 6 small synthetic RGB images in a nested tree."""
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
        img = (rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
        cv2.imwrite(str(p), img)
    return tmp_path


@pytest.fixture
def small_image_uint8() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
```

- [ ] **Step 8: Install with uv and run smoke test**

```bash
uv sync --extra dev
uv run pytest -q
```

Expected: `no tests ran` (no test files yet) — exit code 5 from pytest is acceptable here; verify `uv run coliraz version` prints `0.1.0`.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
coliraz: scaffold project structure

Move DDColor/ to reference/ddcolor_original/, set up uv-managed
pyproject.toml with all runtime+dev deps, create the empty package
tree, add Typer CLI stub with `version` command, and seed tests/
with the image-fixture conftest.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Color conversion utilities (`utils/color.py`)

**Files:**
- Create: `src/coliraz/utils/color.py`
- Create: `tests/test_color.py`

Vectorized RGB ↔ LAB on torch tensors (used everywhere — model input, loss context, preview). Cross-checked against cv2 reference.

- [ ] **Step 1: Write the failing test `tests/test_color.py`**

```python
import cv2
import numpy as np
import torch

from coliraz.utils.color import rgb_to_lab, lab_to_rgb, derive_gray_rgb_from_rgb


def _cv2_rgb_to_lab(rgb_uint8):
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)


def test_rgb_to_lab_matches_cv2_within_tolerance(small_image_uint8):
    rgb_uint8 = small_image_uint8
    rgb_f32 = rgb_uint8.astype(np.float32) / 255.0

    expected = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2LAB)  # H,W,3
    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    got = rgb_to_lab(t).squeeze(0).permute(1, 2, 0).numpy()

    np.testing.assert_allclose(got, expected, atol=1.0)


def test_lab_to_rgb_round_trip(small_image_uint8):
    rgb_f32 = small_image_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)
    lab = rgb_to_lab(t)
    back = lab_to_rgb(lab).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(back, rgb_f32, atol=0.02)


def test_derive_gray_rgb_matches_reference(small_image_uint8):
    rgb_f32 = small_image_uint8.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2LAB)
    L = lab[:, :, :1]
    gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
    expected = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)

    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)
    got = derive_gray_rgb_from_rgb(t).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, expected, atol=0.01)
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_color.py -q
```

Expected: `ImportError: cannot import name 'rgb_to_lab' from 'coliraz.utils.color'`.

- [ ] **Step 3: Implement `src/coliraz/utils/color.py`**

```python
"""Vectorized RGB <-> LAB conversion on torch tensors.

Conventions match cv2's `COLOR_RGB2LAB` for float32 inputs in [0, 1]:
- L in [0, 100]
- a, b in approximately [-128, 127]
All ops are pure-tensor so they autograd through and run on GPU.
"""
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
    threshold = 0.04045
    return torch.where(c <= threshold, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    threshold = 0.0031308
    return torch.where(c <= threshold, c * 12.92, 1.055 * c.clamp(min=0).pow(1 / 2.4) - 0.055)


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(t > delta**3, t.clamp(min=0).pow(1.0 / 3.0), t / (3 * delta**2) + 4.0 / 29.0)


def _f_lab_inv(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(t > delta, t.pow(3), 3 * delta**2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B, 3, H, W) in [0, 1] sRGB → (B, 3, H, W) LAB."""
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(rgb.shape)}")
    m = _RGB2XYZ.to(rgb.device, dtype=rgb.dtype)
    w = _WHITE.to(rgb.device, dtype=rgb.dtype)

    lin = _srgb_to_linear(rgb)
    xyz = torch.einsum("ij,bjhw->bihw", m, lin) / w.view(1, 3, 1, 1)
    f = _f_lab(xyz)
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """lab: (B, 3, H, W) → (B, 3, H, W) sRGB in [0, 1] (may exceed range; clamp at call site)."""
    if lab.dim() != 4 or lab.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(lab.shape)}")
    m = _XYZ2RGB.to(lab.device, dtype=lab.dtype)
    w = _WHITE.to(lab.device, dtype=lab.dtype)

    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = torch.cat([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], dim=1) * w.view(1, 3, 1, 1)
    lin = torch.einsum("ij,bjhw->bihw", m, xyz)
    return _linear_to_srgb(lin)


def derive_gray_rgb_from_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B, 3, H, W) → 3-channel grayscale-as-RGB via LAB-L (model input contract)."""
    lab = rgb_to_lab(rgb)
    L = lab[:, 0:1]
    gray_lab = torch.cat([L, torch.zeros_like(L), torch.zeros_like(L)], dim=1)
    return lab_to_rgb(gray_lab).clamp(0, 1)
```

- [ ] **Step 4: Run tests and verify pass**

```bash
uv run pytest tests/test_color.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/utils/color.py tests/test_color.py
git commit -m "$(cat <<'EOF'
coliraz: add tensor RGB<->LAB color conversion

Pure-torch implementations of rgb_to_lab / lab_to_rgb (D65, sRGB)
and derive_gray_rgb_from_rgb, all batched (B,3,H,W). Cross-checked
against cv2 within atol=1.0 for LAB and 0.02 for the RGB round-trip.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: GPU & timing utilities

**Files:**
- Create: `src/coliraz/utils/gpu.py`
- Create: `src/coliraz/utils/timing.py`
- Create: `tests/test_utils_misc.py`

- [ ] **Step 1: Write the failing test `tests/test_utils_misc.py`**

```python
import time

from coliraz.utils.gpu import gpu_stats, GpuStats
from coliraz.utils.timing import EMA, Stopwatch


def test_gpu_stats_returns_none_or_dataclass():
    s = gpu_stats(device_index=0)
    assert s is None or isinstance(s, GpuStats)


def test_ema_smooths_values():
    ema = EMA(alpha=0.5)
    assert ema.update(10.0) == 10.0
    assert ema.update(20.0) == 15.0
    assert ema.update(20.0) == 17.5


def test_stopwatch_measures_time():
    sw = Stopwatch()
    sw.start()
    time.sleep(0.01)
    sw.stop()
    assert sw.elapsed > 0.005
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_utils_misc.py -q
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/coliraz/utils/gpu.py`**

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

        handle = _HANDLE_CACHE.get(device_index)
        if handle is None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            _HANDLE_CACHE[device_index] = handle
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        try:
            plim = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except Exception:
            plim = 0.0
        return GpuStats(
            name=name,
            mem_used_gb=mem.used / 1024**3,
            mem_total_gb=mem.total / 1024**3,
            util_pct=int(util),
            temp_c=int(temp),
            power_w=pw,
            power_limit_w=plim,
        )
    except Exception:
        return None
```

- [ ] **Step 4: Implement `src/coliraz/utils/timing.py`**

```python
"""Tiny EMA + Stopwatch for the trainer/UI."""
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

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_utils_misc.py -q
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/coliraz/utils/gpu.py src/coliraz/utils/timing.py tests/test_utils_misc.py
git commit -m "$(cat <<'EOF'
coliraz: add gpu_stats and timing utilities

pynvml-backed gpu_stats (returns None if pynvml absent/fails).
EMA accumulator and Stopwatch used by the trainer and Rich UI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Config models, YAML loader, `!preset` tag

**Files:**
- Create: `src/coliraz/config.py`
- Create: `tests/test_config.py`

Pydantic v2 models for the full surface; YAML loader supports `defaults: <file>` chained merging, a `!preset` tag for loss presets, and dict-deep-merge with CLI overrides.

- [ ] **Step 1: Write the failing test `tests/test_config.py`**

```python
from pathlib import Path

import pytest

from coliraz.config import (
    Config,
    LossConfig,
    load_config,
    deep_merge,
    expand_loss_preset,
)


def test_expand_loss_preset_standard():
    losses = expand_loss_preset("standard")
    names = [l.name for l in losses]
    assert "l1_ab" in names
    assert "perceptual_vgg16bn" in names
    assert "colorfulness" in names


def test_expand_loss_preset_minimal():
    losses = expand_loss_preset("minimal")
    assert [l.name for l in losses] == ["l1_ab"]


def test_deep_merge_overrides_leaf_keys():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    over = {"a": {"b": 99}}
    out = deep_merge(base, over)
    assert out == {"a": {"b": 99, "c": 2}, "d": 3}


def test_load_config_with_preset(tmp_path: Path):
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text(
        """
data: { root: /tmp/x }
losses: !preset minimal
"""
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert [l.name for l in cfg.losses] == ["l1_ab"]
    assert cfg.data.root == "/tmp/x"


def test_load_config_chained_defaults(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("data: { root: /a, val_fraction: 0.05 }\n")
    (tmp_path / "child.yaml").write_text(
        "defaults: base.yaml\ndata: { val_fraction: 0.01 }\nlosses: !preset minimal\n"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.data.root == "/a"
    assert cfg.data.val_fraction == 0.01


def test_cli_overrides_win(tmp_path: Path):
    (tmp_path / "x.yaml").write_text("data: { root: /a }\nlosses: !preset minimal\n")
    cfg = load_config(tmp_path / "x.yaml", overrides={"data": {"root": "/b"}})
    assert cfg.data.root == "/b"


def test_required_fields_raise():
    with pytest.raises(Exception):
        # data.root is required, no preset, etc.
        Config.model_validate({})
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_config.py -q
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/coliraz/config.py`**

```python
"""Pydantic v2 config models + YAML loader with chained defaults and !preset tag."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# ---------- model ----------------------------------------------------------

class ModelConfig(BaseModel):
    size: Literal["tiny", "large"] = "tiny"
    input_size: int = 256
    num_queries: int = 100
    num_scales: int = 3
    dec_layers: int = 9
    nf: int = 512
    hidden_dim: int = 256
    refine_norm: Literal["spectral", "batch", "none"] = "spectral"
    encoder_variant: str | None = None  # override timm model name if set


# ---------- data -----------------------------------------------------------

class AugmentConfig(BaseModel):
    hflip: bool = True
    rotate90: bool = False
    color_jitter: bool = False


class LoaderConfig(BaseModel):
    batch_size: int | Literal["auto"] = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


class DataConfig(BaseModel):
    root: str
    val_fraction: float = 0.01
    num_fixed_preview_samples: int = 4
    num_random_preview_samples: int = 2
    augment: AugmentConfig = AugmentConfig()
    loader: LoaderConfig = LoaderConfig()


# ---------- losses ---------------------------------------------------------

class LossConfig(BaseModel):
    name: str
    weight: float = 1.0
    config: dict[str, Any] = Field(default_factory=dict)


_LOSS_PRESETS: dict[str, list[dict[str, Any]]] = {
    "minimal": [{"name": "l1_ab", "weight": 1.0}],
    "standard": [
        {"name": "l1_ab", "weight": 0.1},
        {
            "name": "perceptual_vgg16bn",
            "weight": 5.0,
            "config": {
                "layer_weights": {
                    "conv1_1": 0.0625,
                    "conv2_1": 0.125,
                    "conv3_1": 0.25,
                    "conv4_1": 0.5,
                    "conv5_1": 1.0,
                },
                "criterion": "l1",
            },
        },
        {"name": "colorfulness", "weight": 0.5},
    ],
    "stable": [
        {"name": "charbonnier_ab", "weight": 0.1},
        {"name": "perceptual_vgg16bn", "weight": 5.0, "config": {"criterion": "l1"}},
    ],
    "ddcolor_full": [
        {"name": "l1_ab", "weight": 0.1},
        {
            "name": "perceptual_vgg16bn",
            "weight": 5.0,
            "config": {"criterion": "l1"},
        },
        {
            "name": "gan",
            "weight": 1.0,
            "config": {"gan_type": "hinge", "discriminator": {"type": "unet", "nf": 64}},
        },
        {"name": "colorfulness", "weight": 0.5},
    ],
}


def expand_loss_preset(name: str) -> list[LossConfig]:
    if name not in _LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {name!r}; have {list(_LOSS_PRESETS)}")
    return [LossConfig(**d) for d in _LOSS_PRESETS[name]]


# ---------- optimizer / scheduler -----------------------------------------

class OptimConfig(BaseModel):
    type: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    fused: bool = True


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "multistep", "constant"] = "cosine"
    warmup_steps: int = 2000
    total_steps: int = 400_000
    milestones: list[int] = Field(default_factory=list)
    gamma: float = 0.5


# ---------- training ------------------------------------------------------

class TrainConfig(BaseModel):
    total_steps: int = 400_000
    amp: Literal["bf16", "fp16", "fp32"] = "bf16"
    memory_format: Literal["channels_last", "contiguous"] = "channels_last"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    ema_decay: float = 0.999
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    preview_every_s: float = 10.0
    preview_history_every: int = 10
    ckpt_every_steps: int = 5000
    val_every_steps: int = 5000
    log_every_steps: int = 25
    color_enhance: bool = True
    color_enhance_factor: float = 1.2


class ExportConfig(BaseModel):
    on_finish: bool = True
    opset: int = 17
    simplify: bool = True


class RunConfig(BaseModel):
    name: str = ""
    output_dir: str = ""
    seed: int = 0


# ---------- root ----------------------------------------------------------

class Config(BaseModel):
    run: RunConfig = RunConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig
    losses: list[LossConfig]
    optim_g: OptimConfig = OptimConfig()
    optim_d: OptimConfig = OptimConfig(weight_decay=0.0)
    scheduler: SchedulerConfig = SchedulerConfig()
    train: TrainConfig = TrainConfig()
    export: ExportConfig = ExportConfig()


# ---------- loader --------------------------------------------------------

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
    if "${date:" in value:
        end = value.index("}", value.index("${date:"))
        fmt = value[value.index("${date:") + 7 : end]
        stamp = _dt.datetime.now().strftime(fmt)
        return value.replace(value[value.index("${date:") : end + 1], stamp)
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
    # Interpolate ${date:...} on string leaves once
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            return _interpolate_date(x)
        return x
    raw = walk(raw)
    return Config.model_validate(raw)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_config.py -q
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
coliraz: add Pydantic config models and YAML loader

ModelConfig/DataConfig/LossConfig/TrainConfig etc. with sensible
Blackwell-tuned defaults. YAML loader supports chained 'defaults:',
!preset tag for loss bundles (minimal/standard/stable/ddcolor_full),
deep-merge of CLI overrides, and ${date:...} interpolation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Grayscale pair derivation (`data/grayscale.py`)

**Files:**
- Create: `src/coliraz/data/grayscale.py`
- Create: `tests/test_grayscale.py`

The training/inference image pipeline: RGB uint8 → (RGB-of-gray, GT AB, full-res L). Critical: must match the reference inference path byte-for-byte at typical tolerances.

- [ ] **Step 1: Write the failing test `tests/test_grayscale.py`**

```python
import cv2
import numpy as np

from coliraz.data.grayscale import derive_pair


def test_derive_pair_shapes(small_image_uint8):
    out = derive_pair(small_image_uint8, target_size=16)
    assert out["gray_rgb"].shape == (3, 16, 16)
    assert out["gt_ab"].shape == (2, 16, 16)
    assert out["L_full"].shape == (1, 32, 32)
    assert out["gray_rgb"].dtype.name == "float32"


def test_gray_rgb_matches_reference_pipeline(small_image_uint8):
    """Match the original DDColor ColorizationPipeline grayscale derivation."""
    bgr = cv2.cvtColor(small_image_uint8, cv2.COLOR_RGB2BGR)
    img_f32 = bgr.astype(np.float32) / 255.0
    img_resized = cv2.resize(img_f32, (16, 16))
    L = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
    expected_gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB).transpose(2, 0, 1)

    out = derive_pair(small_image_uint8, target_size=16)
    np.testing.assert_allclose(out["gray_rgb"], expected_gray_rgb, atol=1e-5)
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_grayscale.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/data/grayscale.py`**

```python
"""Grayscale pair derivation used by both training and inference.

Routes RGB through LAB-L exactly like the original ColorizationPipeline,
so training distribution == inference distribution.
"""
from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np


class GrayPair(TypedDict):
    gray_rgb: np.ndarray  # (3, H, W) float32
    gt_ab: np.ndarray  # (2, H, W) float32 — LAB ab channels in cv2 range
    L_full: np.ndarray  # (1, H_full, W_full) float32 — full-res L for inference re-merge


def derive_pair(rgb_uint8: np.ndarray, *, target_size: int) -> GrayPair:
    """rgb_uint8: (H, W, 3) RGB → resized (3xtarget_size^2) gray RGB + GT AB + full-res L."""
    if rgb_uint8.dtype != np.uint8 or rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8 RGB, got {rgb_uint8.shape}/{rgb_uint8.dtype}")

    # cv2 expects BGR
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    img_f32 = bgr.astype(np.float32) / 255.0

    # full-res L for inference re-merge path consistency
    L_full = cv2.cvtColor(img_f32, cv2.COLOR_BGR2Lab)[:, :, :1]  # (H, W, 1)

    img_resized = cv2.resize(img_f32, (target_size, target_size))
    lab_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)  # (T, T, 3)
    L = lab_resized[:, :, :1]
    gt_ab = lab_resized[:, :, 1:].astype(np.float32)

    gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    return GrayPair(
        gray_rgb=gray_rgb.transpose(2, 0, 1),
        gt_ab=gt_ab.transpose(2, 0, 1),
        L_full=L_full.transpose(2, 0, 1),
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_grayscale.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/data/grayscale.py tests/test_grayscale.py
git commit -m "$(cat <<'EOF'
coliraz: add grayscale pair derivation matching original pipeline

derive_pair() routes RGB through LAB to produce the 3-channel
'gray-as-RGB' input the model expects, plus the GT AB target and
the full-res L for the inference re-merge step. Cross-checked
against the reference cv2 routing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Recursive image dataset (`data/dataset.py`)

**Files:**
- Create: `src/coliraz/data/dataset.py`
- Create: `src/coliraz/data/transforms.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test `tests/test_dataset.py`**

```python
from pathlib import Path

import torch

from coliraz.data.dataset import RecursiveImageDataset, build_manifest, MANIFEST_NAME


def test_build_manifest_finds_all_images(tmp_image_dir: Path):
    paths = build_manifest(tmp_image_dir)
    assert len(paths) == 6
    rels = sorted(str(p.relative_to(tmp_image_dir)) for p in paths)
    assert "img0.png" in rels
    assert "a/b/img4.jpeg" in rels


def test_manifest_is_cached(tmp_image_dir: Path):
    build_manifest(tmp_image_dir)
    assert (tmp_image_dir / MANIFEST_NAME).exists()


def test_dataset_returns_correct_shapes(tmp_image_dir: Path):
    ds = RecursiveImageDataset(tmp_image_dir, target_size=32, augment_hflip=False)
    sample = ds[0]
    assert sample["gray_rgb"].shape == (3, 32, 32)
    assert sample["gt_ab"].shape == (2, 32, 32)
    assert sample["L_full"].shape[0] == 1
    assert isinstance(sample["gray_rgb"], torch.Tensor)


def test_dataset_skips_too_small_images(tmp_path: Path):
    import cv2
    import numpy as np
    cv2.imwrite(str(tmp_path / "ok.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "tiny.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    ds = RecursiveImageDataset(tmp_path, target_size=32, min_side=32, augment_hflip=False)
    assert len(ds) == 1


def test_holdout_split_is_deterministic(tmp_image_dir: Path):
    a = RecursiveImageDataset(tmp_image_dir, target_size=32, val_fraction=0.34, split="val", augment_hflip=False)
    b = RecursiveImageDataset(tmp_image_dir, target_size=32, val_fraction=0.34, split="val", augment_hflip=False)
    assert [str(p) for p in a._paths] == [str(p) for p in b._paths]
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_dataset.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/data/transforms.py`**

```python
"""Tiny transforms for the training pipeline (numpy-side, before tensorization)."""
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


def center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return rgb[top : top + size, left : left + size]


def hflip(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, ::-1])
```

- [ ] **Step 4: Implement `src/coliraz/data/dataset.py`**

```python
"""Recursive image dataset with manifest cache and deterministic train/val split."""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .grayscale import derive_pair
from .transforms import center_crop, hflip, random_crop

MANIFEST_NAME = ".coliraz-manifest.txt"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def build_manifest(root: Path, *, force: bool = False) -> list[Path]:
    root = Path(root)
    mf = _manifest_path(root)
    if mf.exists() and not force:
        try:
            mtime_recorded = float(mf.read_text().splitlines()[0])
            if abs(mtime_recorded - root.stat().st_mtime) < 1.0:
                return [root / line for line in mf.read_text().splitlines()[1:]]
        except Exception:
            pass

    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in _EXTS and p.is_file():
            out.append(p)
    try:
        mf.write_text(
            f"{root.stat().st_mtime}\n" + "\n".join(str(p.relative_to(root)) for p in out)
        )
    except OSError:
        pass
    return out


def _hash_to_unit(path: Path) -> float:
    h = hashlib.md5(str(path).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


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

        all_paths = build_manifest(self.root)
        kept: list[Path] = []
        for p in all_paths:
            try:
                # cheap size check via cv2 (header read only via cv2.imread of header)
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is None or im.shape[0] < self.min_side or im.shape[1] < self.min_side:
                    continue
                kept.append(p)
            except Exception:
                continue
        if val_fraction > 0 and split != "all":
            wanted = "val" if split == "val" else "train"
            kept = [
                p for p in kept
                if ((_hash_to_unit(p) < val_fraction) == (wanted == "val"))
            ]
        self._paths = kept

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
        pair = derive_pair(rgb, target_size=self.target_size)
        return {
            "gray_rgb": torch.from_numpy(pair["gray_rgb"]),
            "gt_ab": torch.from_numpy(pair["gt_ab"]),
            "L_full": torch.from_numpy(pair["L_full"]),
            "path": str(p),
        }


def collate(batch: list[dict]) -> dict:
    out = {
        "gray_rgb": torch.stack([b["gray_rgb"] for b in batch]),
        "gt_ab": torch.stack([b["gt_ab"] for b in batch]),
        "L_full": torch.stack([b["L_full"] for b in batch]),
        "path": [b["path"] for b in batch],
    }
    return out
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_dataset.py -q
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/coliraz/data/dataset.py src/coliraz/data/transforms.py tests/test_dataset.py
git commit -m "$(cat <<'EOF'
coliraz: recursive dataset with manifest cache and deterministic split

RecursiveImageDataset walks a directory tree for jpg/png/webp/bmp/tiff,
caches the manifest to .coliraz-manifest.txt (re-scans on root mtime
change), skips images below min_side, and supports deterministic
train/val split via path-hash. Includes random/center crop and hflip
transforms plus a collate helper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: timm-backed ConvNeXt encoder (`models/encoder.py`)

**Files:**
- Create: `src/coliraz/models/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 1: Write the failing test `tests/test_encoder.py`**

```python
import pytest
import torch

from coliraz.models.encoder import ConvNeXtEncoder


@pytest.mark.parametrize("size,expected_channels", [
    ("tiny",  [96, 192, 384, 768]),
    ("large", [192, 384, 768, 1536]),
])
def test_encoder_returns_four_features(size, expected_channels):
    enc = ConvNeXtEncoder(size=size, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    feats = enc(x)
    assert len(feats) == 4
    assert [f.shape[1] for f in feats] == expected_channels
    # spatial dims should halve at each stage (roughly)
    spat = [f.shape[2] for f in feats]
    assert spat[0] > spat[1] > spat[2] > spat[3]


def test_encoder_feature_channels_property():
    enc = ConvNeXtEncoder(size="tiny", pretrained=False)
    assert enc.feature_channels == [96, 192, 384, 768]
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_encoder.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/models/encoder.py`**

```python
"""timm ConvNeXt wrapper exposing multi-scale features."""
from __future__ import annotations

import timm
import torch
from torch import nn

_DEFAULT_VARIANTS = {
    "tiny": "convnext_tiny.fb_in22k",
    "large": "convnext_large.fb_in22k",
}


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        *,
        size: str = "tiny",
        pretrained: bool = True,
        variant: str | None = None,
    ) -> None:
        super().__init__()
        name = variant or _DEFAULT_VARIANTS[size]
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.feature_channels = list(self.backbone.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_encoder.py -q
```

Expected: 3 passed (no network needed because `pretrained=False`).

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/models/encoder.py tests/test_encoder.py
git commit -m "$(cat <<'EOF'
coliraz: add timm ConvNeXt encoder wrapper

ConvNeXtEncoder thin wrapper over timm.create_model with
features_only=True returning four multi-scale stages.
Defaults to convnext_tiny.fb_in22k / convnext_large.fb_in22k
with overridable variant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: UNet blocks (`models/unet_blocks.py`)

**Files:**
- Create: `src/coliraz/models/unet_blocks.py`
- Create: `tests/test_unet_blocks.py`

Modernized port of the original `CustomPixelShuffle_ICNR` and `UnetBlockWide`. The ICNR initialization preserves the original behavior for sub-pixel convolution; the spectral-norm path uses `nn.utils.parametrizations.spectral_norm`.

- [ ] **Step 1: Write the failing test `tests/test_unet_blocks.py`**

```python
import torch

from coliraz.models.unet_blocks import PixelShuffleICNR, UnetBlockWide


def test_pixel_shuffle_icnr_doubles_resolution():
    blk = PixelShuffleICNR(in_ch=32, out_ch=16, scale=2)
    x = torch.randn(2, 32, 8, 8)
    y = blk(x)
    assert y.shape == (2, 16, 16, 16)


def test_pixel_shuffle_icnr_scale4():
    blk = PixelShuffleICNR(in_ch=64, out_ch=32, scale=4)
    x = torch.randn(1, 64, 4, 4)
    y = blk(x)
    assert y.shape == (1, 32, 16, 16)


def test_unet_block_wide_shapes():
    blk = UnetBlockWide(in_c=128, skip_c=64, out_c=96)
    deep = torch.randn(2, 128, 8, 8)
    skip = torch.randn(2, 64, 16, 16)
    y = blk(deep, skip)
    assert y.shape == (2, 96, 16, 16)
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_unet_blocks.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/models/unet_blocks.py`**

```python
"""Pixel-shuffle ICNR upsampler and UNet wide block (modernized)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _icnr_init(tensor: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Initialize a sub-pixel conv weight to mimic nearest-neighbor upsampling at start."""
    out_c, in_c, kh, kw = tensor.shape
    sub_out = out_c // (scale * scale)
    sub = torch.empty(sub_out, in_c, kh, kw)
    nn.init.kaiming_normal_(sub)
    sub = sub.repeat_interleave(scale * scale, dim=0)
    with torch.no_grad():
        tensor.copy_(sub)
    return tensor


class PixelShuffleICNR(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, scale: int = 2, blur: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, kernel_size=1)
        _icnr_init(self.conv.weight, scale=scale)
        self.shuf = nn.PixelShuffle(scale)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.blur = (
            nn.Sequential(nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1))
            if blur else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.shuf(x)
        x = self.blur(x)
        x = self.norm(x)
        return self.act(x)


class UnetBlockWide(nn.Module):
    """Upsample deep feature, concat with skip feature from encoder, project to out_c."""

    def __init__(self, in_c: int, skip_c: int, out_c: int, *, use_spectral: bool = True) -> None:
        super().__init__()
        self.up = PixelShuffleICNR(in_c, in_c // 2, scale=2)
        conv = nn.Conv2d(in_c // 2 + skip_c, out_c, kernel_size=3, padding=1)
        self.proj = spectral_norm(conv) if use_spectral else conv
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(deep)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_unet_blocks.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/models/unet_blocks.py tests/test_unet_blocks.py
git commit -m "$(cat <<'EOF'
coliraz: PixelShuffleICNR upsampler and UnetBlockWide

Modernized port of the original sub-pixel upsampler with ICNR init
and the wide UNet decoder block, using parametrizations.spectral_norm
(compile-safe). Includes optional blur and resolution-correction
interpolation for asymmetric skip dims.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Pixel decoder (`models/pixel_decoder.py`)

**Files:**
- Create: `src/coliraz/models/pixel_decoder.py`
- Create: `tests/test_pixel_decoder.py`

Takes encoder's 4-stage features, produces 3 mid-scale memory features for the color decoder plus 1 hi-res feature map for the einsum step.

- [ ] **Step 1: Write the failing test `tests/test_pixel_decoder.py`**

```python
import torch

from coliraz.models.pixel_decoder import PixelDecoder


def test_pixel_decoder_outputs():
    feature_channels = [96, 192, 384, 768]  # tiny
    dec = PixelDecoder(feature_channels=feature_channels, nf=512)
    # mock encoder features at 64x64 input → /4, /8, /16, /32
    feats = [
        torch.randn(2, 96, 16, 16),
        torch.randn(2, 192, 8, 8),
        torch.randn(2, 384, 4, 4),
        torch.randn(2, 768, 2, 2),
    ]
    mem, hi = dec(feats)
    assert len(mem) == 3
    # hi should be ~/1 (4x of /4 stage)
    assert hi.shape[2] == 64 and hi.shape[3] == 64
    assert hi.shape[1] == 256
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_pixel_decoder.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/models/pixel_decoder.py`**

```python
"""Pixel decoder: UNet upsample path producing 3 mid-scale memory features + 1 hi-res map."""
from __future__ import annotations

import torch
from torch import nn

from .unet_blocks import PixelShuffleICNR, UnetBlockWide


class PixelDecoder(nn.Module):
    def __init__(self, *, feature_channels: list[int], nf: int = 512) -> None:
        super().__init__()
        c0, c1, c2, c3 = feature_channels
        out_c = nf
        # u1: deep=c3, skip=c2 -> out_c
        self.u1 = UnetBlockWide(in_c=c3, skip_c=c2, out_c=out_c)
        # u2: deep=out_c, skip=c1 -> out_c
        self.u2 = UnetBlockWide(in_c=out_c, skip_c=c1, out_c=out_c)
        # u3: deep=out_c, skip=c0 -> out_c // 2
        self.u3 = UnetBlockWide(in_c=out_c, skip_c=c0, out_c=out_c // 2)
        # final 4x upsample for hi-res feature map (color decoder einsum operand)
        self.last_shuf = PixelShuffleICNR(out_c // 2, out_c // 2, scale=4)

    def forward(self, feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        f0, f1, f2, f3 = feats
        m0 = self.u1(f3, f2)   # /16-ish at out_c
        m1 = self.u2(m0, f1)   # /8 at out_c
        m2 = self.u3(m1, f0)   # /4 at out_c//2
        hi = self.last_shuf(m2)  # ~/1 at out_c//2
        return [m0, m1, m2], hi
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pixel_decoder.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/models/pixel_decoder.py tests/test_pixel_decoder.py
git commit -m "$(cat <<'EOF'
coliraz: pixel decoder (3-stage UNet + 4x pixel-shuffle)

PixelDecoder consumes the encoder's 4 multi-scale features, runs
three UnetBlockWide upsamples, and emits 3 mid-scale memory
tensors (consumed by the color decoder as transformer memory)
plus 1 hi-res feature map for the final einsum projection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Color decoder transformer (`models/color_decoder.py`)

**Files:**
- Create: `src/coliraz/models/color_decoder.py`
- Create: `tests/test_color_decoder.py`

Multi-scale transformer with learnable color queries. Uses `nn.MultiheadAttention(batch_first=True)` so SDPA / FlashAttention is auto-selected on supported GPUs.

- [ ] **Step 1: Write the failing test `tests/test_color_decoder.py`**

```python
import torch

from coliraz.models.color_decoder import MultiScaleColorDecoder


def test_color_decoder_einsum_output():
    in_chs = [512, 512, 256]
    dec = MultiScaleColorDecoder(
        in_channels=in_chs, num_queries=8, num_scales=3, dec_layers=2, hidden_dim=64
    )
    memories = [
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 512, 8, 8),
        torch.randn(2, 256, 4, 4),
    ]
    hi = torch.randn(2, 64, 64, 64)
    out = dec(memories, hi)
    assert out.shape == (2, 8, 64, 64)
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_color_decoder.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/models/color_decoder.py`**

```python
"""Transformer color decoder with learnable color queries (SDPA-backed)."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinePositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats: int = 32, temperature: int = 10000) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — only shape is used
        b, _, h, w = x.shape
        y_embed = torch.arange(1, h + 1, device=x.device, dtype=x.dtype)[None, :, None].repeat(b, 1, w)
        x_embed = torch.arange(1, w + 1, device=x.device, dtype=x.dtype)[None, None, :].repeat(b, h, 1)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


class _TransformerDecoderLayer(nn.Module):
    """Cross-attn → self-attn → FFN, all batch_first."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, mem: torch.Tensor, query_pos: torch.Tensor, mem_pos: torch.Tensor) -> torch.Tensor:
        # Cross
        q_pe = query + query_pos
        k_pe = mem + mem_pos
        out, _ = self.cross_attn(q_pe, k_pe, mem, need_weights=False)
        query = self.norm1(query + out)
        # Self
        q_pe = query + query_pos
        out, _ = self.self_attn(q_pe, q_pe, query, need_weights=False)
        query = self.norm2(query + out)
        # FFN
        out = self.ffn(query)
        return self.norm3(query + out)


class _MLP(nn.Module):
    def __init__(self, in_d: int, hid: int, out_d: int, n: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_d
        for _ in range(n - 1):
            layers += [nn.Linear(prev, hid), nn.ReLU(inplace=True)]
            prev = hid
        layers.append(nn.Linear(prev, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiScaleColorDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: list[int],
        num_queries: int = 100,
        hidden_dim: int = 256,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        num_scales: int = 3,
        color_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        assert len(in_channels) == num_scales
        self.num_queries = num_queries
        self.num_scales = num_scales
        self.dec_layers = dec_layers

        self.pe = SinePositionalEncoding(num_pos_feats=hidden_dim // 2)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_scales, hidden_dim)

        self.input_proj = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_channels]
        )
        for p in self.input_proj:
            nn.init.kaiming_uniform_(p.weight, a=1)
            if p.bias is not None:
                nn.init.constant_(p.bias, 0)

        self.layers = nn.ModuleList(
            [
                _TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward)
                for _ in range(dec_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.color_embed = _MLP(hidden_dim, hidden_dim, color_embed_dim, n=3)

    def forward(self, memories: list[torch.Tensor], hi_feat: torch.Tensor) -> torch.Tensor:
        b = hi_feat.shape[0]
        src_seq: list[torch.Tensor] = []
        pos_seq: list[torch.Tensor] = []
        for i, m in enumerate(memories):
            proj = self.input_proj[i](m)
            pos = self.pe(proj)
            level = self.level_embed.weight[i].view(1, -1, 1, 1)
            s = (proj + level).flatten(2).transpose(1, 2)  # (B, HW, C)
            p = pos.flatten(2).transpose(1, 2)
            src_seq.append(s)
            pos_seq.append(p)

        query = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1)
        q_pe = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        for i, layer in enumerate(self.layers):
            level = i % self.num_scales
            query = layer(query, src_seq[level], q_pe, pos_seq[level])

        query = self.norm(query)
        emb = self.color_embed(query)  # (B, Q, color_embed_dim)
        out = torch.einsum("bqc,bchw->bqhw", emb, hi_feat)
        return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_color_decoder.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/models/color_decoder.py tests/test_color_decoder.py
git commit -m "$(cat <<'EOF'
coliraz: multi-scale transformer color decoder (SDPA)

Modernized DETR-style decoder: learnable color queries cross-attend
to 3 mid-scale pixel features, then the final per-query embeddings
project to per-pixel color via einsum. Uses nn.MultiheadAttention
(batch_first) so the backend selects FlashAttention on Blackwell.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Top-level DDColor model (`models/refine.py`, `models/ddcolor.py`)

**Files:**
- Create: `src/coliraz/models/refine.py`
- Create: `src/coliraz/models/ddcolor.py`
- Modify: `src/coliraz/models/__init__.py`
- Create: `tests/test_ddcolor.py`

- [ ] **Step 1: Write the failing test `tests/test_ddcolor.py`**

```python
import torch

from coliraz.models.ddcolor import build_ddcolor
from coliraz.config import ModelConfig


def test_ddcolor_tiny_forward_shape():
    cfg = ModelConfig(size="tiny", input_size=64, dec_layers=2, num_queries=8)
    model = build_ddcolor(cfg, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 2, 64, 64)


def test_ddcolor_backward_flows():
    cfg = ModelConfig(size="tiny", input_size=64, dec_layers=2, num_queries=8)
    model = build_ddcolor(cfg, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    loss = model(x).pow(2).mean()
    loss.backward()
    # at least one parameter received a grad
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_ddcolor.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/models/refine.py`**

```python
"""Final 1x1 refinement conv that mixes the einsum coarse map with the input RGB."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def build_refine(in_ch: int, out_ch: int, *, norm: str = "spectral") -> nn.Module:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
    if norm == "spectral":
        return spectral_norm(conv)
    if norm == "batch":
        return nn.Sequential(conv, nn.BatchNorm2d(out_ch))
    if norm == "none":
        return conv
    raise ValueError(f"unknown refine norm: {norm!r}")
```

- [ ] **Step 4: Implement `src/coliraz/models/ddcolor.py`**

```python
"""Top-level DDColor module wiring encoder → pixel decoder → color decoder → refine."""
from __future__ import annotations

import torch
from torch import nn

from coliraz.config import ModelConfig

from .color_decoder import MultiScaleColorDecoder
from .encoder import ConvNeXtEncoder
from .pixel_decoder import PixelDecoder
from .refine import build_refine


class DDColor(nn.Module):
    def __init__(self, cfg: ModelConfig, *, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvNeXtEncoder(
            size=cfg.size, pretrained=pretrained, variant=cfg.encoder_variant
        )
        self.pixel_decoder = PixelDecoder(
            feature_channels=self.encoder.feature_channels, nf=cfg.nf
        )
        hi_ch = cfg.nf // 2
        memory_chs = [cfg.nf, cfg.nf, cfg.nf // 2]
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=memory_chs,
            num_queries=cfg.num_queries,
            hidden_dim=cfg.hidden_dim,
            dec_layers=cfg.dec_layers,
            num_scales=cfg.num_scales,
            color_embed_dim=hi_ch,
        )
        self.refine = build_refine(cfg.num_queries + 3, 2, norm=cfg.refine_norm)

        # ImageNet normalization buffers (model input is RGB in [0, 1])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize for the ImageNet-pretrained encoder
        x_norm = (x - self.mean) / self.std
        feats = self.encoder(x_norm)
        memories, hi = self.pixel_decoder(feats)
        coarse = self.color_decoder(memories, hi)
        # input RGB-of-gray is concatenated for the refinement step
        if coarse.shape[-2:] != x.shape[-2:]:
            coarse = torch.nn.functional.interpolate(
                coarse, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.refine(torch.cat([coarse, x_norm], dim=1))


def build_ddcolor(cfg: ModelConfig, *, pretrained: bool = True) -> DDColor:
    return DDColor(cfg, pretrained=pretrained)
```

- [ ] **Step 5: Update `src/coliraz/models/__init__.py`**

```python
from .ddcolor import DDColor, build_ddcolor

__all__ = ["DDColor", "build_ddcolor"]
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_ddcolor.py -q
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/coliraz/models/refine.py src/coliraz/models/ddcolor.py \
        src/coliraz/models/__init__.py tests/test_ddcolor.py
git commit -m "$(cat <<'EOF'
coliraz: top-level DDColor model wiring all components

build_ddcolor(cfg) composes ConvNeXt encoder, pixel decoder, color
decoder, and a 1x1 spectral-norm refine into a single nn.Module.
Input contract: (B, 3, H, W) RGB-of-gray in [0,1]; output: (B, 2, H, W) AB.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: UNet discriminator (`models/discriminator.py`)

**Files:**
- Create: `src/coliraz/models/discriminator.py`
- Create: `tests/test_discriminator.py`

- [ ] **Step 1: Write the failing test `tests/test_discriminator.py`**

```python
import torch

from coliraz.models.discriminator import UNetDiscriminator


def test_discriminator_forward_shape():
    d = UNetDiscriminator(in_ch=3, nf=16)
    x = torch.randn(2, 3, 64, 64)
    y = d(x)
    # Returns a per-pixel logit map at input resolution
    assert y.shape == (2, 1, 64, 64)
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_discriminator.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/models/discriminator.py`**

```python
"""UNet-style image discriminator with per-pixel logits."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _down(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(in_c: int, out_c: int) -> nn.Sequential:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        u3 = self.u3(d4)
        u2 = self.u2(torch.cat([u3, d3], dim=1))
        u1 = self.u1(torch.cat([u2, d2], dim=1))
        y = self.out(torch.cat([u1, d1], dim=1))
        return self.up_final(y)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_discriminator.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/models/discriminator.py tests/test_discriminator.py
git commit -m "$(cat <<'EOF'
coliraz: UNet-style image discriminator with per-pixel logits

Spectral-norm down/up path producing a single-channel logit map at
input resolution — consumed only when the GAN loss is enabled.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Loss registry & LossContext (`losses/registry.py`)

**Files:**
- Create: `src/coliraz/losses/registry.py`
- Create: `tests/test_loss_registry.py`

- [ ] **Step 1: Write the failing test `tests/test_loss_registry.py`**

```python
import torch

from coliraz.losses.registry import (
    LossContext,
    ColorizationLoss,
    LOSS_REGISTRY,
    register_loss,
    build_loss,
)


def test_registry_collects_decorated_class():
    @register_loss("toy_test_loss")
    class _Toy(ColorizationLoss):
        name = "toy_test_loss"
        def forward(self, ctx):
            return ctx.pred_ab.abs().mean()

    assert "toy_test_loss" in LOSS_REGISTRY
    LOSS_REGISTRY.pop("toy_test_loss")  # cleanup


def test_build_loss_returns_module_with_name():
    @register_loss("toy_build")
    class _Toy(ColorizationLoss):
        name = "toy_build"
        def __init__(self, k: int = 1):
            super().__init__()
            self.k = k
        def forward(self, ctx):
            return ctx.pred_ab.abs().mean() * self.k

    m = build_loss("toy_build", {"k": 3})
    ctx = LossContext(
        pred_ab=torch.zeros(1, 2, 4, 4) + 1.0,
        gt_ab=torch.zeros(1, 2, 4, 4),
        pred_rgb=torch.zeros(1, 3, 4, 4),
        gt_rgb=torch.zeros(1, 3, 4, 4),
        gray_rgb=torch.zeros(1, 3, 4, 4),
    )
    assert m(ctx).item() == 3.0
    LOSS_REGISTRY.pop("toy_build")
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_loss_registry.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/losses/registry.py`**

```python
"""Loss registry, LossContext dataclass, and a tiny build_loss factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import torch
from torch import nn


@dataclass
class LossContext:
    pred_ab: torch.Tensor
    gt_ab: torch.Tensor
    pred_rgb: torch.Tensor
    gt_rgb: torch.Tensor
    gray_rgb: torch.Tensor
    discriminator: nn.Module | None = None


class ColorizationLoss(nn.Module):
    """Base class for all coloration losses. Subclasses must set `name`."""
    name: str = ""

    def forward(self, ctx: LossContext) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


LOSS_REGISTRY: dict[str, Type[ColorizationLoss]] = {}


def register_loss(name: str):
    def deco(cls: Type[ColorizationLoss]):
        if name in LOSS_REGISTRY:
            raise KeyError(f"loss {name!r} already registered")
        cls.name = name
        LOSS_REGISTRY[name] = cls
        return cls
    return deco


def build_loss(name: str, cfg: dict[str, Any] | None = None) -> ColorizationLoss:
    if name not in LOSS_REGISTRY:
        raise KeyError(f"unknown loss {name!r}; have {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**(cfg or {}))
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_loss_registry.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/losses/registry.py tests/test_loss_registry.py
git commit -m "$(cat <<'EOF'
coliraz: loss registry, LossContext, build_loss factory

@register_loss decorator + class-level name field; build_loss(name, cfg)
instantiates a loss with kwargs. LossContext carries the shared per-step
state every loss might need.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Pixel losses (`losses/pixel.py`)

**Files:**
- Create: `src/coliraz/losses/pixel.py`
- Create: `tests/test_pixel_losses.py`

- [ ] **Step 1: Write the failing test `tests/test_pixel_losses.py`**

```python
import torch

from coliraz.losses.pixel import L1AbLoss, L2AbLoss, CharbonnierAbLoss
from coliraz.losses.registry import LossContext


def _ctx(pred=None, gt=None):
    z = torch.zeros(1, 3, 4, 4)
    return LossContext(
        pred_ab=pred if pred is not None else torch.zeros(1, 2, 4, 4),
        gt_ab=gt if gt is not None else torch.zeros(1, 2, 4, 4),
        pred_rgb=z, gt_rgb=z, gray_rgb=z,
    )


def test_l1_zero_when_equal():
    assert L1AbLoss()(_ctx()).item() == 0.0


def test_l1_positive_when_unequal():
    pred = torch.zeros(1, 2, 4, 4) + 1.0
    assert L1AbLoss()(_ctx(pred=pred)).item() == 1.0


def test_l2_positive_when_unequal():
    pred = torch.zeros(1, 2, 4, 4) + 2.0
    assert L2AbLoss()(_ctx(pred=pred)).item() == 4.0


def test_charbonnier_grad():
    pred = torch.randn(1, 2, 4, 4, requires_grad=True)
    out = CharbonnierAbLoss()(_ctx(pred=pred))
    out.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_pixel_losses.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/losses/pixel.py`**

```python
"""Pixel-space losses (operate on the predicted AB channels)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import ColorizationLoss, LossContext, register_loss


@register_loss("l1_ab")
class L1AbLoss(ColorizationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.l1_loss(ctx.pred_ab, ctx.gt_ab)


@register_loss("l2_ab")
class L2AbLoss(ColorizationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.mse_loss(ctx.pred_ab, ctx.gt_ab)


@register_loss("charbonnier_ab")
class CharbonnierAbLoss(ColorizationLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        diff2 = (ctx.pred_ab - ctx.gt_ab) ** 2
        return torch.sqrt(diff2 + self.eps2).mean()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pixel_losses.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/losses/pixel.py tests/test_pixel_losses.py
git commit -m "$(cat <<'EOF'
coliraz: pixel losses (L1/L2/Charbonnier on AB)

Three registered pixel losses, each receives LossContext and returns
a scalar reduction over the predicted AB channels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Perceptual loss (`losses/perceptual.py`)

**Files:**
- Create: `src/coliraz/losses/perceptual.py`
- Create: `tests/test_perceptual_loss.py`

VGG16-bn perceptual loss with lazy weight load. Test uses a stubbed VGG to keep CI fast & offline.

- [ ] **Step 1: Write the failing test `tests/test_perceptual_loss.py`**

```python
import torch
from torch import nn

from coliraz.losses.perceptual import VGG16BNPerceptualLoss
from coliraz.losses.registry import LossContext


def _stub_vgg_layers() -> dict[str, nn.Module]:
    return {
        "conv1_1": nn.Conv2d(3, 4, 3, padding=1),
        "conv2_1": nn.Conv2d(4, 8, 3, padding=1),
        "conv3_1": nn.Conv2d(8, 16, 3, padding=1),
        "conv4_1": nn.Conv2d(16, 16, 3, padding=1),
        "conv5_1": nn.Conv2d(16, 16, 3, padding=1),
    }


def test_perceptual_loss_grad_flows():
    loss = VGG16BNPerceptualLoss.__new__(VGG16BNPerceptualLoss)
    nn.Module.__init__(loss)
    stub = _stub_vgg_layers()
    loss._stages = nn.ModuleDict(stub)
    loss._weights = {k: 1.0 for k in stub}
    loss._criterion = nn.L1Loss()
    loss.style_weight = 0.0
    loss._input_norm = True
    loss.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    loss.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    pred = torch.randn(1, 3, 16, 16, requires_grad=True)
    gt = torch.randn(1, 3, 16, 16)
    z = torch.zeros(1, 2, 16, 16)
    ctx = LossContext(pred_ab=z, gt_ab=z, pred_rgb=pred, gt_rgb=gt, gray_rgb=pred)
    out = loss(ctx)
    out.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_perceptual_loss.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/losses/perceptual.py`**

```python
"""VGG16-BN perceptual loss with lazy weight load."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from .registry import ColorizationLoss, LossContext, register_loss

# vgg16_bn conv layer indices used by the original DDColor config
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
class VGG16BNPerceptualLoss(ColorizationLoss):
    def __init__(
        self,
        layer_weights: Mapping[str, float] | None = None,
        criterion: str = "l1",
        style_weight: float = 0.0,
        use_input_norm: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import VGG16_BN_Weights, vgg16_bn

        if layer_weights is None:
            layer_weights = {
                "conv1_1": 0.0625,
                "conv2_1": 0.125,
                "conv3_1": 0.25,
                "conv4_1": 0.5,
                "conv5_1": 1.0,
            }
        self._weights = dict(layer_weights)
        self.style_weight = float(style_weight)
        self._input_norm = bool(use_input_norm)

        vgg_features = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        vgg_features.train(False)
        for p in vgg_features.parameters():
            p.requires_grad_(False)

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        last = 0
        for name in sorted(self._weights, key=lambda k: _LAYER_INDICES[k]):
            idx = _LAYER_INDICES[name]
            stages[name] = nn.Sequential(*list(vgg_features[last : idx + 1]))
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
        pred_f = self._features(ctx.pred_rgb)
        with torch.no_grad():
            gt_f = self._features(ctx.gt_rgb)
        perc = 0.0
        for name, w in self._weights.items():
            perc = perc + w * self._criterion(pred_f[name], gt_f[name].detach())
        if self.style_weight > 0:
            sty = 0.0
            for name, w in self._weights.items():
                sty = sty + w * self._criterion(_gram(pred_f[name]), _gram(gt_f[name].detach()))
            perc = perc + self.style_weight * sty
        return perc
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_perceptual_loss.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/losses/perceptual.py tests/test_perceptual_loss.py
git commit -m "$(cat <<'EOF'
coliraz: VGG16-BN perceptual loss with optional style term

Layer-weighted VGG16-BN feature L1/L2 with optional Gram-matrix style
component. VGG weights downloaded lazily on first instantiation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: GAN losses (`losses/gan.py`)

**Files:**
- Create: `src/coliraz/losses/gan.py`
- Create: `tests/test_gan_loss.py`

- [ ] **Step 1: Write the failing test `tests/test_gan_loss.py`**

```python
import torch
from torch import nn

from coliraz.losses.gan import GeneratorGANLoss, discriminator_loss
from coliraz.losses.registry import LossContext


class _TinyDisc(nn.Module):
    def forward(self, x):
        return x.mean(dim=1, keepdim=True)


def _ctx(disc):
    z2 = torch.zeros(1, 2, 8, 8)
    rgb = torch.randn(1, 3, 8, 8, requires_grad=True)
    gt = torch.randn(1, 3, 8, 8)
    return LossContext(
        pred_ab=z2, gt_ab=z2, pred_rgb=rgb, gt_rgb=gt, gray_rgb=rgb, discriminator=disc,
    ), rgb


def test_generator_gan_loss_hinge_grad():
    disc = _TinyDisc()
    loss = GeneratorGANLoss(gan_type="hinge")
    ctx, rgb = _ctx(disc)
    out = loss(ctx)
    out.backward()
    assert rgb.grad is not None


def test_discriminator_loss_returns_scalar():
    disc = _TinyDisc()
    real = torch.randn(1, 3, 8, 8)
    fake = torch.randn(1, 3, 8, 8)
    d = discriminator_loss(disc, real, fake, gan_type="hinge")
    assert d.dim() == 0
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_gan_loss.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/losses/gan.py`**

```python
"""GAN losses (generator-side via registry; discriminator step as a helper)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .registry import ColorizationLoss, LossContext, register_loss


def _g_loss(logits: torch.Tensor, gan_type: str) -> torch.Tensor:
    if gan_type == "vanilla":
        return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    if gan_type == "lsgan":
        return F.mse_loss(logits, torch.ones_like(logits))
    if gan_type == "hinge":
        return -logits.mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


def _d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, gan_type: str) -> torch.Tensor:
    if gan_type == "vanilla":
        r = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        f = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return r + f
    if gan_type == "lsgan":
        return F.mse_loss(real_logits, torch.ones_like(real_logits)) + F.mse_loss(
            fake_logits, torch.zeros_like(fake_logits)
        )
    if gan_type == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


@register_loss("gan")
class GeneratorGANLoss(ColorizationLoss):
    def __init__(self, gan_type: str = "hinge", discriminator: dict | None = None) -> None:
        super().__init__()
        self.gan_type = gan_type
        self._disc_cfg = discriminator or {"type": "unet", "nf": 64}

    @property
    def disc_config(self) -> dict:
        return self._disc_cfg

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.discriminator is None:
            raise RuntimeError("GeneratorGANLoss requires LossContext.discriminator")
        fake_logits = ctx.discriminator(ctx.pred_rgb)
        return _g_loss(fake_logits, self.gan_type)


def discriminator_loss(disc: nn.Module, real_rgb: torch.Tensor, fake_rgb: torch.Tensor, gan_type: str) -> torch.Tensor:
    real_logits = disc(real_rgb)
    fake_logits = disc(fake_rgb.detach())
    return _d_loss(real_logits, fake_logits, gan_type)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_gan_loss.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/losses/gan.py tests/test_gan_loss.py
git commit -m "$(cat <<'EOF'
coliraz: GAN losses (vanilla/lsgan/hinge) + discriminator helper

GeneratorGANLoss reads ctx.discriminator and returns generator-side
scalar. discriminator_loss(disc, real, fake) helper used by the
trainer's separate D-step. Discriminator architecture cfg is carried
as metadata so the trainer can build it once at startup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Colorfulness loss (`losses/colorfulness.py`)

**Files:**
- Create: `src/coliraz/losses/colorfulness.py`
- Create: `tests/test_colorfulness_loss.py`

- [ ] **Step 1: Write the failing test `tests/test_colorfulness_loss.py`**

```python
import torch

from coliraz.losses.colorfulness import ColorfulnessLoss
from coliraz.losses.registry import LossContext


def test_colorfulness_decreases_with_more_color():
    z = torch.zeros(1, 2, 4, 4)
    gray = torch.zeros(1, 3, 4, 4) + 0.5
    color = torch.zeros(1, 3, 4, 4); color[0, 0] = 1.0
    ctx_gray = LossContext(pred_ab=z, gt_ab=z, pred_rgb=gray, gt_rgb=gray, gray_rgb=gray)
    ctx_color = LossContext(pred_ab=z, gt_ab=z, pred_rgb=color, gt_rgb=color, gray_rgb=gray)
    loss = ColorfulnessLoss()
    assert loss(ctx_color) < loss(ctx_gray)


def test_colorfulness_grad():
    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    z = torch.zeros(1, 2, 4, 4)
    ctx = LossContext(pred_ab=z, gt_ab=z, pred_rgb=rgb, gt_rgb=rgb, gray_rgb=rgb)
    ColorfulnessLoss()(ctx).backward()
    assert rgb.grad is not None
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_colorfulness_loss.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/losses/colorfulness.py`**

```python
"""Colorfulness loss — negated Hasler & Susstrunk colorfulness metric.

We want to *maximize* colorfulness, so the loss is its negation.
"""
from __future__ import annotations

import torch

from .registry import ColorizationLoss, LossContext, register_loss


@register_loss("colorfulness")
class ColorfulnessLoss(ColorizationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        rgb = ctx.pred_rgb.clamp(0, 1)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        sigma = torch.sqrt(rg.var(dim=(1, 2)) + yb.var(dim=(1, 2)) + 1e-8)
        mu = torch.sqrt(rg.mean(dim=(1, 2)) ** 2 + yb.mean(dim=(1, 2)) ** 2 + 1e-8)
        colorfulness = sigma + 0.3 * mu
        return -colorfulness.mean()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_colorfulness_loss.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/losses/colorfulness.py tests/test_colorfulness_loss.py
git commit -m "$(cat <<'EOF'
coliraz: colorfulness loss (negated Hasler-Susstrunk metric)

Rewards higher chroma in the predicted RGB. Off by default; included
in the standard/ddcolor_full presets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: LossSet composer (`losses/__init__.py`)

**Files:**
- Modify: `src/coliraz/losses/__init__.py`
- Create: `tests/test_loss_set.py`

- [ ] **Step 1: Write the failing test `tests/test_loss_set.py`**

```python
import torch

from coliraz.config import LossConfig
from coliraz.losses import LossSet
from coliraz.losses.registry import LossContext


def test_loss_set_aggregates_with_weights():
    ls = LossSet([LossConfig(name="l1_ab", weight=2.0)])
    pred = torch.zeros(1, 2, 4, 4) + 1.0
    z = torch.zeros(1, 2, 4, 4)
    rgb = torch.zeros(1, 3, 4, 4)
    ctx = LossContext(pred_ab=pred, gt_ab=z, pred_rgb=rgb, gt_rgb=rgb, gray_rgb=rgb)
    total, log = ls(ctx)
    assert total.item() == 2.0
    assert log["l1_ab"] == 1.0


def test_loss_set_has_gan_detected():
    with_gan = LossSet([
        LossConfig(name="l1_ab", weight=1.0),
        LossConfig(name="gan", weight=1.0, config={"gan_type": "hinge"}),
    ])
    without_gan = LossSet([LossConfig(name="l1_ab", weight=1.0)])
    assert with_gan.has_gan is True
    assert without_gan.has_gan is False


def test_loss_set_disc_cfg_only_if_gan():
    ls = LossSet([
        LossConfig(name="gan", weight=1.0, config={"gan_type": "hinge", "discriminator": {"type": "unet", "nf": 32}}),
    ])
    assert ls.discriminator_cfg == {"type": "unet", "nf": 32}
```

- [ ] **Step 2: Implement `src/coliraz/losses/__init__.py`**

```python
"""Loss aggregation."""
from __future__ import annotations

import torch

from coliraz.config import LossConfig

# Importing the submodules registers the losses
from . import pixel as _pixel  # noqa: F401
from . import perceptual as _perceptual  # noqa: F401
from . import gan as _gan  # noqa: F401
from . import colorfulness as _colorfulness  # noqa: F401
from .registry import LossContext, build_loss
from .gan import GeneratorGANLoss


class LossSet:
    """Composes a list of weighted losses; aggregates totals and emits a flat log dict."""

    def __init__(self, configs: list[LossConfig]):
        self.entries: list[tuple[float, object]] = []
        self.has_gan = False
        self.discriminator_cfg: dict | None = None
        for c in configs:
            loss = build_loss(c.name, c.config)
            self.entries.append((float(c.weight), loss))
            if isinstance(loss, GeneratorGANLoss):
                self.has_gan = True
                self.discriminator_cfg = loss.disc_config

    def parameters(self):
        for _, loss in self.entries:
            for p in loss.parameters():
                yield p

    def to(self, device, dtype=None):
        for _, loss in self.entries:
            loss.to(device, dtype) if dtype is not None else loss.to(device)
        return self

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        total: torch.Tensor | float = 0.0
        log: dict[str, float] = {}
        for weight, loss in self.entries:
            val = loss(ctx)
            total = total + weight * val
            log[loss.name] = float(val.detach())
        if isinstance(total, float):
            total = torch.zeros((), device=ctx.pred_ab.device)
        return total, log


__all__ = ["LossSet", "LossContext", "build_loss"]
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_loss_set.py -q
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add src/coliraz/losses/__init__.py tests/test_loss_set.py
git commit -m "$(cat <<'EOF'
coliraz: LossSet composer aggregating weighted losses

Reads a list of LossConfig, builds each, exposes has_gan flag and
discriminator_cfg (consumed by the trainer to build the D module
exactly once). Importing the package registers every shipped loss.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: EMA helper (`train/ema.py`)

**Files:**
- Create: `src/coliraz/train/ema.py`
- Create: `tests/test_ema_module.py`

- [ ] **Step 1: Write the failing test `tests/test_ema_module.py`**

```python
import torch
from torch import nn

from coliraz.train.ema import ModelEMA


def test_ema_converges_to_model_after_many_updates():
    m = nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0); m.bias.fill_(0.0)
    for _ in range(20):
        ema.update(m)
    assert torch.allclose(ema.module.weight, m.weight, atol=1e-3)


def test_ema_state_dict_round_trip():
    m = nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.9)
    sd = ema.state_dict()
    ema2 = ModelEMA(nn.Linear(2, 2), decay=0.9)
    ema2.load_state_dict(sd)
    for k in sd:
        assert torch.equal(ema2.state_dict()[k], sd[k])
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_ema_module.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/train/ema.py`**

```python
"""Fp32 EMA wrapper for the model."""
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

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_ema_module.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/train/ema.py tests/test_ema_module.py
git commit -m "$(cat <<'EOF'
coliraz: fp32 ModelEMA tracker

Maintains a deep-copied fp32 shadow of the model that the preview
writer and final checkpoint consume. Skips non-float buffers verbatim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Checkpointing (`train/checkpoint.py`)

**Files:**
- Create: `src/coliraz/train/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing test `tests/test_checkpoint.py`**

```python
import torch
from torch import nn

from coliraz.train.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_round_trip(tmp_path):
    m = nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=m, optimizer=opt, step=42, extra={"foo": "bar"})

    m2 = nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    payload = load_checkpoint(path, model=m2, optimizer=opt2)
    assert payload["step"] == 42
    assert payload["extra"]["foo"] == "bar"
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p.data, q.data)
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_checkpoint.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/train/checkpoint.py`**

```python
"""Simple checkpoint save/load with atomic write."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    ema=None,
    scheduler=None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    *,
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

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_checkpoint.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/train/checkpoint.py tests/test_checkpoint.py
git commit -m "$(cat <<'EOF'
coliraz: atomic checkpoint save/load

Single payload dict covering model, both optimizers, discriminator,
EMA, scheduler and arbitrary extra. Atomic write via .tmp + os.replace.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Preview writer (`train/preview.py`)

**Files:**
- Create: `src/coliraz/train/preview.py`
- Create: `tests/test_preview.py`

- [ ] **Step 1: Write the failing test `tests/test_preview.py`**

```python
import numpy as np
import torch

from coliraz.train.preview import render_preview_grid, write_png_atomic


def test_render_preview_grid_returns_uint8_image():
    samples = []
    for _ in range(3):
        samples.append(
            {
                "original": torch.rand(3, 32, 32),
                "gray_rgb": torch.rand(3, 32, 32),
                "pred_rgb": torch.rand(3, 32, 32),
                "delta_ab": torch.rand(2, 32, 32),
            }
        )
    img = render_preview_grid(samples, caption="step 100", cell_size=32)
    assert img.dtype == np.uint8
    assert img.shape[1] == 32 * 4
    assert img.shape[2] == 3


def test_preview_write_atomic(tmp_path):
    img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    write_png_atomic(p, img)
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_preview.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/train/preview.py`**

```python
"""Render preview comparison grid: [original | gray | pred | |delta_ab|]."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch


def _t_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).round().astype(np.uint8)


def _delta_ab_to_heatmap(delta_ab: torch.Tensor, max_val: float = 50.0) -> np.ndarray:
    mag = torch.linalg.vector_norm(delta_ab.detach().float().cpu(), dim=0)
    mag = (mag.clamp(0, max_val) / max_val * 255.0).to(torch.uint8).numpy()
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def render_preview_grid(
    samples: list[dict[str, torch.Tensor]],
    *,
    caption: str,
    cell_size: int = 256,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for s in samples:
        tiles = []
        for key in ("original", "gray_rgb", "pred_rgb"):
            img = _t_to_uint8_rgb(s[key])
            if img.shape[:2] != (cell_size, cell_size):
                img = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            tiles.append(img)
        delta = _delta_ab_to_heatmap(s["delta_ab"])
        if delta.shape[:2] != (cell_size, cell_size):
            delta = cv2.resize(delta, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        rows.append(np.concatenate(tiles + [delta], axis=1))
    body = np.concatenate(rows, axis=0)

    cap_h = 24
    cap = np.zeros((cap_h, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(cap, caption, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([cap, body], axis=0)


def write_png_atomic(path: str | Path, img_rgb_uint8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_preview.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/train/preview.py tests/test_preview.py
git commit -m "$(cat <<'EOF'
coliraz: preview grid renderer + atomic PNG writer

render_preview_grid composes a 4-column comparison
(original | gray | predicted | |Δ ab| heatmap) with a caption strip.
write_png_atomic uses .tmp + os.replace so external watchers never
see a half-written file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: Rich live UI (`train/ui.py`)

**Files:**
- Create: `src/coliraz/train/ui.py`
- Create: `tests/test_ui_smoke.py`

- [ ] **Step 1: Write the failing test `tests/test_ui_smoke.py`**

```python
from coliraz.train.ui import TrainUI


def test_ui_can_render_one_frame():
    ui = TrainUI(run_name="t", total_steps=100, headless=True)
    ui.tick(step=1, losses={"l1_ab": 0.5}, lr=1e-4, throughput_imgs=10.0)
    ui.tick(step=2, losses={"l1_ab": 0.4}, lr=1e-4, throughput_imgs=12.0)
    frame = ui.render()
    assert frame is not None
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_ui_smoke.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/train/ui.py`**

```python
"""Rich live dashboard for training."""
from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from coliraz.utils.gpu import gpu_stats
from coliraz.utils.timing import EMA


@dataclass
class _LossTrack:
    ema: EMA = field(default_factory=lambda: EMA(alpha=0.1))
    long_ema: EMA = field(default_factory=lambda: EMA(alpha=0.01))


class TrainUI(AbstractContextManager):
    def __init__(self, *, run_name: str, total_steps: int, headless: bool = False) -> None:
        self.run_name = run_name
        self.total_steps = total_steps
        self.headless = headless
        self.console = Console()
        self._losses: dict[str, _LossTrack] = {}
        self._lr: float = 0.0
        self._throughput_imgs: float = 0.0
        self._step: int = 0
        self._last_preview: str = ""
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

    def tick(self, *, step: int, losses: dict[str, float], lr: float, throughput_imgs: float) -> None:
        self._step = step
        self._lr = lr
        self._throughput_imgs = throughput_imgs
        for k, v in losses.items():
            t = self._losses.setdefault(k, _LossTrack())
            t.ema.update(v)
            t.long_ema.update(v)
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
            Layout(Panel.fit(f"run: {self.run_name}", title="coliraz train"), size=3),
            Layout(self._progress, size=3),
            Layout(name="middle", size=14),
            Layout(Panel.fit(self._last_preview or "(no preview yet)", title="last preview"), size=3),
        )
        layout["middle"].split_row(self._losses_panel(), self._gpu_panel())
        return layout

    def _losses_panel(self) -> Panel:
        t = Table.grid(padding=(0, 1))
        t.add_column("loss")
        t.add_column("value", justify="right")
        t.add_column("trend", justify="right")
        for name, tr in self._losses.items():
            ema = tr.ema.value or 0.0
            longer = tr.long_ema.value or ema
            arrow = "▼" if ema < longer else "▲"
            t.add_row(name, f"{ema:.4f}", f"{arrow} {abs(ema - longer):.4f}")
        t.add_row("lr", f"{self._lr:.2e}", "")
        t.add_row("img/s", f"{self._throughput_imgs:.1f}", "")
        return Panel(t, title="losses (EMA)")

    def _gpu_panel(self) -> Panel:
        s = gpu_stats(0)
        if s is None:
            return Panel("gpu stats unavailable", title="gpu")
        t = Table.grid(padding=(0, 1))
        t.add_column(); t.add_column(justify="right")
        t.add_row("name", s.name)
        t.add_row("mem", f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GB")
        t.add_row("util", f"{s.util_pct}%")
        t.add_row("temp", f"{s.temp_c}°C")
        t.add_row("pwr", f"{s.power_w:.0f}/{s.power_limit_w:.0f} W")
        return Panel(t, title="gpu")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_ui_smoke.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/train/ui.py tests/test_ui_smoke.py
git commit -m "$(cat <<'EOF'
coliraz: Rich live training dashboard

Three-pane layout: progress bar (with ETA), losses panel (EMA-smoothed
+ trend arrow), and GPU stats panel (skips gracefully if pynvml absent).
headless=True path is used by tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: Trainer (`train/trainer.py`)

**Files:**
- Create: `src/coliraz/train/trainer.py`
- Modify: `src/coliraz/train/__init__.py`
- Create: `tests/test_train_step.py`

The single class wiring data, model, losses, AMP, EMA, UI, and preview together. The test overfits a single batch on CPU using the tiny model and asserts the L1 loss decreases.

- [ ] **Step 1: Write the failing test `tests/test_train_step.py`**

```python
import torch

from coliraz.config import (
    Config, DataConfig, LoaderConfig, LossConfig, ModelConfig,
    OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
)
from coliraz.train.trainer import Trainer


def _make_cfg(tmp_path):
    return Config(
        run=RunConfig(name="t", output_dir=str(tmp_path), seed=0),
        model=ModelConfig(size="tiny", input_size=64, dec_layers=1, num_queries=4, nf=64, hidden_dim=32),
        data=DataConfig(
            root=str(tmp_path),
            loader=LoaderConfig(batch_size=2, num_workers=0, persistent_workers=False),
        ),
        losses=[LossConfig(name="l1_ab", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=10),
        train=TrainConfig(
            total_steps=10, amp="fp32", memory_format="contiguous",
            compile=False, ema_decay=0.0, preview_every_s=10000,
            ckpt_every_steps=10000, log_every_steps=1,
        ),
    )


def test_trainer_reduces_loss_on_overfit(tmp_image_dir, tmp_path):
    cfg = _make_cfg(tmp_image_dir)
    cfg.run.output_dir = str(tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"), pretrained_encoder=False, headless=True)
    initial = trainer.run_one_step()
    for _ in range(20):
        last = trainer.run_one_step()
    assert last["total_g"] < initial["total_g"]
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_train_step.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/train/trainer.py`**

```python
"""Single-class trainer wiring everything together."""
from __future__ import annotations

import itertools
import math
import threading
import time
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from coliraz.config import Config, LossConfig
from coliraz.data.dataset import RecursiveImageDataset, collate
from coliraz.losses import LossSet, LossContext
from coliraz.losses.gan import discriminator_loss
from coliraz.models import build_ddcolor
from coliraz.models.discriminator import UNetDiscriminator
from coliraz.utils.color import lab_to_rgb, rgb_to_lab

from .checkpoint import save_checkpoint
from .ema import ModelEMA
from .preview import render_preview_grid, write_png_atomic
from .ui import TrainUI


def _build_optimizer(model_params, cfg) -> torch.optim.Optimizer:
    klass = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam, "SGD": torch.optim.SGD}[cfg.type]
    kw = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if cfg.type != "SGD":
        kw["betas"] = tuple(cfg.betas)
        try:
            kw["fused"] = cfg.fused
        except TypeError:
            pass
    try:
        return klass(model_params, **kw)
    except (TypeError, ValueError):
        kw.pop("fused", None)
        return klass(model_params, **kw)


def _build_scheduler(opt, cfg, total_steps: int):
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
    def __init__(
        self,
        cfg: Config,
        *,
        device: torch.device | None = None,
        pretrained_encoder: bool = True,
        headless: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.run.seed)

        # Model
        memory_format = torch.channels_last if cfg.train.memory_format == "channels_last" else torch.contiguous_format
        self.model = build_ddcolor(cfg.model, pretrained=pretrained_encoder).to(self.device, memory_format=memory_format)
        self.memory_format = memory_format

        # Optimizer
        self.opt_g = _build_optimizer(self.model.parameters(), cfg.optim_g)
        self.scheduler_g = _build_scheduler(self.opt_g, cfg.scheduler, cfg.train.total_steps)

        # Losses
        self.loss_set = LossSet(cfg.losses)
        for _, loss in self.loss_set.entries:
            loss.to(self.device)
        self.disc: nn.Module | None = None
        self.opt_d: torch.optim.Optimizer | None = None
        if self.loss_set.has_gan:
            dcfg = self.loss_set.discriminator_cfg or {}
            self.disc = UNetDiscriminator(in_ch=3, nf=int(dcfg.get("nf", 64))).to(self.device)
            self.opt_d = _build_optimizer(self.disc.parameters(), cfg.optim_d)
            self.gan_type = "hinge"
            for _, loss_mod in self.loss_set.entries:
                if hasattr(loss_mod, "gan_type"):
                    self.gan_type = loss_mod.gan_type
                    break

        # EMA
        self.ema = ModelEMA(self.model, decay=cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None

        # AMP
        amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
        self.amp_dtype = amp_map[cfg.train.amp]
        self.scaler = torch.amp.GradScaler("cuda") if (cfg.train.amp == "fp16" and self.device.type == "cuda") else None

        # Data
        ds = RecursiveImageDataset(
            cfg.data.root,
            target_size=cfg.model.input_size,
            val_fraction=cfg.data.val_fraction,
            split="train" if cfg.data.val_fraction > 0 else "all",
            augment_hflip=cfg.data.augment.hflip,
            augment_rotate90=cfg.data.augment.rotate90,
            seed=cfg.run.seed,
        )
        if len(ds) == 0:
            raise RuntimeError(f"no images under {cfg.data.root}")
        bs = cfg.data.loader.batch_size if cfg.data.loader.batch_size != "auto" else 16
        self.train_loader = DataLoader(
            ds,
            batch_size=int(bs),
            shuffle=True,
            num_workers=cfg.data.loader.num_workers,
            pin_memory=cfg.data.loader.pin_memory and self.device.type == "cuda",
            persistent_workers=cfg.data.loader.persistent_workers and cfg.data.loader.num_workers > 0,
            prefetch_factor=cfg.data.loader.prefetch_factor if cfg.data.loader.num_workers > 0 else None,
            collate_fn=collate,
            drop_last=True,
        )

        # Val/preview dataset
        if cfg.data.val_fraction > 0:
            self.val_ds = RecursiveImageDataset(
                cfg.data.root,
                target_size=cfg.model.input_size,
                val_fraction=cfg.data.val_fraction,
                split="val",
                augment_hflip=False,
                seed=cfg.run.seed,
            )
        else:
            self.val_ds = ds

        # UI
        self.ui = TrainUI(run_name=cfg.run.name, total_steps=cfg.train.total_steps, headless=headless)
        self._iter = _cycle(self.train_loader)
        self.step = 0
        self._last_preview_t = 0.0
        self._t_window = time.perf_counter()
        self._samples_window = 0
        self._preview_lock = threading.Lock()

    # ------------------------------------------------------------------
    # one step (used by tests + main loop)
    # ------------------------------------------------------------------
    def run_one_step(self) -> dict[str, float]:
        batch = next(self._iter)
        return self._train_step(batch)

    def _train_step(self, batch: dict) -> dict[str, float]:
        gray_rgb = batch["gray_rgb"].to(self.device, non_blocking=True, memory_format=self.memory_format)
        gt_ab = batch["gt_ab"].to(self.device, non_blocking=True)
        L = batch["L_full"].to(self.device, non_blocking=True)

        self.opt_g.zero_grad(set_to_none=True)
        amp_ctx = torch.amp.autocast(self.device.type, dtype=self.amp_dtype) if self.amp_dtype is not None else _noop_ctx()
        with amp_ctx:
            pred_ab = self.model(gray_rgb)
            pred_lab = torch.cat([L, pred_ab], dim=1)
            pred_rgb = lab_to_rgb(pred_lab).clamp(0, 1)
            gt_lab = torch.cat([L, gt_ab], dim=1)
            gt_rgb = lab_to_rgb(gt_lab).clamp(0, 1)
            ctx = LossContext(
                pred_ab=pred_ab, gt_ab=gt_ab,
                pred_rgb=pred_rgb, gt_rgb=gt_rgb,
                gray_rgb=gray_rgb, discriminator=self.disc,
            )
            total_g, log_g = self.loss_set(ctx)

        if self.scaler is not None:
            self.scaler.scale(total_g).backward()
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)
            self.scaler.step(self.opt_g); self.scaler.update()
        else:
            total_g.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)
            self.opt_g.step()

        log: dict[str, float] = {"total_g": float(total_g.detach()), **log_g}
        if self.disc is not None and self.opt_d is not None:
            self.opt_d.zero_grad(set_to_none=True)
            with amp_ctx:
                d_loss = discriminator_loss(self.disc, gt_rgb.detach(), pred_rgb.detach(), self.gan_type)
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
        self._samples_window += gray_rgb.shape[0]
        return log

    # ------------------------------------------------------------------
    # full loop
    # ------------------------------------------------------------------
    def fit(self) -> None:
        with self.ui:
            for _ in range(self.cfg.train.total_steps):
                log = self.run_one_step()
                if self.step % self.cfg.train.log_every_steps == 0:
                    now = time.perf_counter()
                    imgs_per_s = self._samples_window / max(1e-6, now - self._t_window)
                    self._t_window = now
                    self._samples_window = 0
                    lr = self.opt_g.param_groups[0]["lr"]
                    self.ui.tick(step=self.step, losses=log, lr=lr, throughput_imgs=imgs_per_s)
                if time.perf_counter() - self._last_preview_t >= self.cfg.train.preview_every_s:
                    self._write_preview()
                if self.step % self.cfg.train.ckpt_every_steps == 0:
                    self._save_ckpt(name="last.pt")
            self._save_ckpt(name="final.pt")
            if self.cfg.export.on_finish:
                self._maybe_export_onnx()

    # ------------------------------------------------------------------
    # preview + ckpt + export helpers
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _build_preview_samples(self) -> list[dict]:
        n_fixed = self.cfg.data.num_fixed_preview_samples
        n_rand = self.cfg.data.num_random_preview_samples
        eval_model = self.ema.module if self.ema is not None else self.model
        eval_model.train(False)
        out: list[dict] = []
        idxs = list(range(min(n_fixed, len(self.val_ds))))
        if n_rand > 0 and len(self.val_ds) > n_fixed:
            extra = list(torch.randint(n_fixed, len(self.val_ds), (n_rand,)).tolist())
            idxs += extra
        for i in idxs:
            s = self.val_ds[i]
            gray = s["gray_rgb"].unsqueeze(0).to(self.device)
            L = s["L_full"].unsqueeze(0).to(self.device)
            gt_ab = s["gt_ab"].unsqueeze(0).to(self.device)
            pred_ab = eval_model(gray)
            pred_lab = torch.cat([L, pred_ab], dim=1)
            pred_rgb = lab_to_rgb(pred_lab).clamp(0, 1).squeeze(0)
            orig_lab = torch.cat([L, gt_ab], dim=1)
            orig_rgb = lab_to_rgb(orig_lab).clamp(0, 1).squeeze(0)
            delta = (pred_ab - gt_ab).squeeze(0)
            out.append({
                "original": orig_rgb,
                "gray_rgb": s["gray_rgb"],
                "pred_rgb": pred_rgb,
                "delta_ab": delta,
            })
        if self.ema is None:
            self.model.train(True)
        return out

    def _write_preview(self) -> None:
        with self._preview_lock:
            samples = self._build_preview_samples()
            caption = f"step {self.step}  ts {time.strftime('%H:%M:%S')}"
            grid = render_preview_grid(samples, caption=caption, cell_size=self.cfg.model.input_size)
            latest = self.output_dir / "samples" / "latest.png"
            write_png_atomic(latest, grid)
            if self.cfg.train.preview_history_every > 0 and (self.step % self.cfg.train.preview_history_every == 0):
                hist = self.output_dir / "samples" / f"iter_{self.step:07d}.png"
                write_png_atomic(hist, grid)
            self.ui.note_preview(f"wrote {latest.relative_to(self.output_dir)} @ step {self.step}")
            self._last_preview_t = time.perf_counter()

    def _save_ckpt(self, name: str) -> None:
        path = self.output_dir / "ckpt" / name
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.opt_g,
            optimizer_d=self.opt_d,
            discriminator=self.disc,
            ema=self.ema,
            scheduler=self.scheduler_g,
            step=self.step,
            extra={"cfg": self.cfg.model_dump()},
        )

    def _maybe_export_onnx(self) -> None:
        try:
            from coliraz.export.onnx import export_onnx_from_model
        except Exception:
            return
        path = self.output_dir / "model.onnx"
        export_onnx_from_model(
            self.ema.module if self.ema is not None else self.model,
            input_size=self.cfg.model.input_size,
            export_path=path,
            opset=self.cfg.export.opset,
            simplify=self.cfg.export.simplify,
        )


class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def fit(cfg: Config, *, device: torch.device | None = None) -> None:
    Trainer(cfg, device=device).fit()
```

- [ ] **Step 4: Update `src/coliraz/train/__init__.py`**

```python
from .trainer import Trainer, fit

__all__ = ["Trainer", "fit"]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_train_step.py -q
```

Expected: 1 passed (loss reduces after 20 overfit steps).

- [ ] **Step 6: Commit**

```bash
git add src/coliraz/train/trainer.py src/coliraz/train/__init__.py tests/test_train_step.py
git commit -m "$(cat <<'EOF'
coliraz: Trainer wires data + model + losses + UI + preview

Single-class trainer with AMP (bf16 default), channels-last memory
format, EMA shadow, optional GAN+discriminator step, cosine LR with
warmup, periodic preview PNG, atomic checkpoint, and optional ONNX
export on finish. headless mode and run_one_step() expose the loop
for tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 24: Inference pipeline (`infer/pipeline.py`)

**Files:**
- Create: `src/coliraz/infer/pipeline.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test `tests/test_inference.py`**

```python
import cv2
import numpy as np
import torch

from coliraz.config import ModelConfig
from coliraz.infer.pipeline import ColorizationPipeline
from coliraz.models import build_ddcolor


def test_pipeline_returns_same_shape_bgr_uint8(tmp_path):
    cfg = ModelConfig(size="tiny", input_size=64, dec_layers=1, num_queries=4, nf=64, hidden_dim=32)
    model = build_ddcolor(cfg, pretrained=False)
    pipe = ColorizationPipeline(model, input_size=64, device=torch.device("cpu"))
    img = (np.random.rand(48, 72, 3) * 255).astype(np.uint8)
    out = pipe.process(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_inference.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/coliraz/infer/pipeline.py`**

```python
"""Inference pipeline: replicates the original ColorizationPipeline LAB routing."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from coliraz.train.checkpoint import load_checkpoint


class ColorizationPipeline:
    def __init__(self, model: nn.Module, *, input_size: int, device: torch.device | None = None) -> None:
        self.input_size = int(input_size)
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.train(False)

    @torch.inference_mode()
    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None:
            raise ValueError("img is None")
        h, w = img_bgr.shape[:2]
        img = img_bgr.astype(np.float32) / 255.0
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        l_low = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        gray_lab = np.concatenate([l_low, np.zeros_like(l_low), np.zeros_like(l_low)], axis=-1)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)

        tensor = (
            torch.from_numpy(gray_rgb.transpose(2, 0, 1))
            .float().unsqueeze(0).to(self.device)
        )
        output_ab = self.model(tensor).cpu()
        output_ab_resized = (
            F.interpolate(output_ab, size=(h, w), mode="bilinear", align_corners=False)[0]
            .float().numpy().transpose(1, 2, 0)
        )
        output_lab = np.concatenate([orig_l, output_ab_resized], axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        return (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)


def load_pipeline(checkpoint: str | Path, *, input_size: int, device: torch.device | None = None) -> ColorizationPipeline:
    from coliraz.config import ModelConfig
    from coliraz.models import build_ddcolor

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg = ModelConfig(**(cfg_dict.get("model") or {"input_size": input_size}))
    model = build_ddcolor(mcfg, pretrained=False)
    model.load_state_dict(payload["model"])
    return ColorizationPipeline(model, input_size=input_size, device=device)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_inference.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/infer/pipeline.py tests/test_inference.py
git commit -m "$(cat <<'EOF'
coliraz: inference pipeline (LAB routing + full-res re-merge)

ColorizationPipeline matches the original byte-for-byte: full-res L
kept aside, model runs at fixed input_size, predicted AB upsampled
bilinearly, re-merged with original L in LAB. load_pipeline reads
the embedded model config from the checkpoint extra dict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 25: ONNX export with parity check (`export/onnx.py`)

**Files:**
- Create: `src/coliraz/export/onnx.py`
- Create: `tests/test_export_onnx.py`

- [ ] **Step 1: Write the failing test `tests/test_export_onnx.py`**

```python
import os

import numpy as np
import pytest
import torch

from coliraz.config import ModelConfig
from coliraz.export.onnx import export_onnx_from_model
from coliraz.models import build_ddcolor


@pytest.mark.skipif(os.environ.get("COLIRAZ_SLOW") != "1",
                    reason="onnx export is slow; set COLIRAZ_SLOW=1 to run")
def test_onnx_export_parity(tmp_path):
    cfg = ModelConfig(size="tiny", input_size=32, dec_layers=1, num_queries=2, nf=64, hidden_dim=32)
    model = build_ddcolor(cfg, pretrained=False)
    model.train(False)
    path = tmp_path / "m.onnx"
    export_onnx_from_model(model, input_size=32, export_path=path, opset=17, simplify=False)
    assert path.exists()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    onnx_out = sess.run(None, {"input": x})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-3, rtol=1e-2)
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_export_onnx.py -q
```

Expected: skipped (no COLIRAZ_SLOW) → 1 skipped, 0 failed.

- [ ] **Step 3: Implement `src/coliraz/export/onnx.py`**

```python
"""Export DDColor to ONNX with shape inference, simplification, and parity check."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def export_onnx_from_model(
    model: nn.Module,
    *,
    input_size: int,
    export_path: str | Path,
    opset: int = 17,
    simplify: bool = True,
    verify_parity: bool = True,
    parity_atol: float = 1e-3,
) -> None:
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.train(False)

    dummy = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(export_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    if simplify:
        try:
            import onnx
            import onnxsim
            m = onnx.load(str(export_path))
            m, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m, str(export_path))
        except Exception:
            pass

    if verify_parity:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
        ort_out = sess.run(None, {"input": x})[0]
        with torch.no_grad():
            t_out = model(torch.from_numpy(x)).numpy()
        diff = float(np.abs(ort_out - t_out).max())
        if diff > parity_atol:
            raise RuntimeError(f"ONNX parity failed: max_abs_diff={diff:.3e} > atol={parity_atol}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_export_onnx.py -q
COLIRAZ_SLOW=1 uv run pytest tests/test_export_onnx.py -q
```

Expected: first run: 1 skipped. Second run: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/coliraz/export/onnx.py tests/test_export_onnx.py
git commit -m "$(cat <<'EOF'
coliraz: ONNX export with parity verification

export_onnx_from_model writes opset-17 ONNX (dynamic batch),
optionally simplifies via onnxsim, then runs the exported model
under onnxruntime and asserts max_abs_diff <= 1e-3 against the
PyTorch reference output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 26: Default + tiny + large configs

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/tiny.yaml`
- Create: `configs/large.yaml`
- Create: `tests/test_configs_load.py`

- [ ] **Step 1: Write the failing test `tests/test_configs_load.py`**

```python
from pathlib import Path

from coliraz.config import load_config

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_tiny_yaml_loads():
    cfg = load_config(ROOT / "tiny.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "tiny"
    assert cfg.data.root == "/tmp"


def test_large_yaml_loads():
    cfg = load_config(ROOT / "large.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "large"
```

- [ ] **Step 2: Run and verify failure**

```bash
uv run pytest tests/test_configs_load.py -q
```

Expected: FileNotFoundError.

- [ ] **Step 3: Create `configs/default.yaml`**

```yaml
run:
  name: "${date:%Y-%m-%d_%H-%M-%S}"
  output_dir: "runs/${date:%Y-%m-%d_%H-%M-%S}"
  seed: 0

model:
  size: tiny
  input_size: 256
  num_queries: 100
  num_scales: 3
  dec_layers: 9
  nf: 512
  hidden_dim: 256
  refine_norm: spectral

data:
  val_fraction: 0.01
  num_fixed_preview_samples: 4
  num_random_preview_samples: 2
  augment:
    hflip: true
    rotate90: false
    color_jitter: false
  loader:
    batch_size: 32
    num_workers: 16
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4

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
  total_steps: 400000

train:
  total_steps: 400000
  amp: bf16
  memory_format: channels_last
  compile: false
  compile_mode: default
  ema_decay: 0.999
  grad_accum_steps: 1
  clip_grad_norm: 1.0
  preview_every_s: 10.0
  preview_history_every: 10
  ckpt_every_steps: 5000
  val_every_steps: 5000
  log_every_steps: 25
  color_enhance: true
  color_enhance_factor: 1.2

export:
  on_finish: true
  opset: 17
  simplify: true
```

- [ ] **Step 4: Create `configs/tiny.yaml`**

```yaml
defaults: default.yaml
model:
  size: tiny
data:
  loader:
    batch_size: 64
```

- [ ] **Step 5: Create `configs/large.yaml`**

```yaml
defaults: default.yaml
model:
  size: large
data:
  loader:
    batch_size: 16
losses: !preset ddcolor_full
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_configs_load.py -q
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add configs/default.yaml configs/tiny.yaml configs/large.yaml tests/test_configs_load.py
git commit -m "$(cat <<'EOF'
coliraz: default + tiny + large config presets

default.yaml carries the full surface with Blackwell-tuned defaults.
tiny.yaml/large.yaml override just model.size and batch_size; large
also switches to the ddcolor_full loss preset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 27: CLI commands (`cli.py`)

**Files:**
- Modify: `src/coliraz/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test `tests/test_cli.py`**

```python
from typer.testing import CliRunner

from coliraz.cli import app

runner = CliRunner()


def test_help_top_level():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    out = r.stdout
    assert "train" in out and "infer" in out and "export" in out and "scan-data" in out


def test_help_train():
    r = runner.invoke(app, ["train", "--help"])
    assert r.exit_code == 0


def test_scan_data_writes_manifest(tmp_image_dir):
    r = runner.invoke(app, ["scan-data", "--root", str(tmp_image_dir)])
    assert r.exit_code == 0
    assert (tmp_image_dir / ".coliraz-manifest.txt").exists()
```

- [ ] **Step 2: Implement `src/coliraz/cli.py` (replace prior stub)**

```python
"""coliraz CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="coliraz — image colorization", no_args_is_help=True)


@app.command()
def version() -> None:
    from coliraz import __version__
    typer.echo(__version__)


@app.command(name="scan-data")
def scan_data(root: Path = typer.Option(..., "--root", exists=True, file_okay=False)) -> None:
    """Build/refresh the recursive image manifest under ROOT."""
    from coliraz.data.dataset import build_manifest

    paths = build_manifest(root, force=True)
    typer.echo(f"{len(paths)} images indexed under {root}")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    data: Optional[Path] = typer.Option(None, "--data", help="override data.root"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="override run.name"),
    batch_size: Optional[str] = typer.Option(None, "--batch-size"),
    compile_: bool = typer.Option(False, "--compile/--no-compile", help="torch.compile model"),
    amp: Optional[str] = typer.Option(None, "--amp", help="bf16|fp16|fp32"),
    total_steps: Optional[int] = typer.Option(None, "--total-steps"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
) -> None:
    """Train a colorization model from images recursively rooted at --data."""
    import torch

    from coliraz.config import load_config
    from coliraz.train import Trainer
    from coliraz.train.checkpoint import load_checkpoint

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
        load_checkpoint(
            resume,
            model=trainer.model,
            optimizer=trainer.opt_g,
            optimizer_d=trainer.opt_d,
            discriminator=trainer.disc,
            ema=trainer.ema,
            scheduler=trainer.scheduler_g,
        )
    trainer.fit()


@app.command()
def infer(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    input_: Path = typer.Option(..., "--input", "--in", exists=True),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(512, "--input-size"),
    batch: int = typer.Option(1, "--batch"),
) -> None:
    """Colorize a single image or a directory."""
    import cv2
    import torch

    from coliraz.infer.pipeline import load_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(model, input_size=input_size, device=device)
    if input_.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(input_))
        cv2.imwrite(str(output), pipe.process(img))
    else:
        output.mkdir(parents=True, exist_ok=True)
        for p in sorted(input_.rglob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            out_path = output / p.relative_to(input_)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), pipe.process(img))
    typer.echo(f"wrote {output}")


@app.command()
def export(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(512, "--input-size"),
    opset: int = typer.Option(17, "--opset"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify"),
) -> None:
    """Export a checkpoint to ONNX with parity verification."""
    import torch

    from coliraz.config import ModelConfig
    from coliraz.export.onnx import export_onnx_from_model
    from coliraz.models import build_ddcolor

    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**((payload.get("extra") or {}).get("cfg", {}).get("model") or {"input_size": input_size}))
    m = build_ddcolor(mcfg, pretrained=False)
    m.load_state_dict(payload["model"])
    export_onnx_from_model(m, input_size=input_size, export_path=output, opset=opset, simplify=simplify)
    typer.echo(f"wrote {output}")
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_cli.py -q
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add src/coliraz/cli.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
coliraz: full Typer CLI (train, infer, export, scan-data)

Each subcommand is a typed function. train accepts inline overrides
that merge into the YAML config (--data/--batch-size/--amp/etc.).
infer handles single image or recursive directory; export reads the
model cfg from the checkpoint extra dict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 28: End-to-end smoke test

**Files:**
- Create: `tests/test_e2e_smoke.py`

Final verification: the user-visible path — `coliraz train` on a synthetic image tree — runs for a handful of steps, writes a preview PNG and a checkpoint, then `coliraz infer` colorizes one image.

- [ ] **Step 1: Write the test `tests/test_e2e_smoke.py`**

```python
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from coliraz.config import (
    Config, DataConfig, ExportConfig, LoaderConfig, LossConfig, ModelConfig,
    OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
)
from coliraz.train import Trainer
from coliraz.infer.pipeline import load_pipeline


@pytest.mark.skipif(os.environ.get("COLIRAZ_SLOW") != "1",
                    reason="end-to-end smoke is slow; set COLIRAZ_SLOW=1 to run")
def test_train_then_infer_e2e(tmp_path):
    data_dir = tmp_path / "imgs"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        img = (rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8))
        cv2.imwrite(str(data_dir / f"img{i}.png"), img)

    out_dir = tmp_path / "run"
    cfg = Config(
        run=RunConfig(name="smoke", output_dir=str(out_dir), seed=0),
        model=ModelConfig(size="tiny", input_size=64, dec_layers=1, num_queries=4, nf=64, hidden_dim=32),
        data=DataConfig(
            root=str(data_dir),
            val_fraction=0.25,
            num_fixed_preview_samples=1,
            num_random_preview_samples=0,
            loader=LoaderConfig(batch_size=2, num_workers=0, persistent_workers=False),
        ),
        losses=[LossConfig(name="l1_ab", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=5),
        train=TrainConfig(
            total_steps=5, amp="fp32", memory_format="contiguous", compile=False,
            ema_decay=0.0, preview_every_s=0.0, preview_history_every=0,
            ckpt_every_steps=5, log_every_steps=1,
        ),
        export=ExportConfig(on_finish=False),
    )
    trainer = Trainer(cfg, device=torch.device("cpu"), pretrained_encoder=False, headless=True)
    trainer.fit()

    final_ckpt = out_dir / "ckpt" / "final.pt"
    assert final_ckpt.exists()
    assert (out_dir / "samples" / "latest.png").exists()

    pipe = load_pipeline(final_ckpt, input_size=64, device=torch.device("cpu"))
    img = cv2.imread(str(data_dir / "img0.png"))
    out = pipe.process(img)
    assert out.shape == img.shape and out.dtype == np.uint8
```

- [ ] **Step 2: Run with COLIRAZ_SLOW=1**

```bash
COLIRAZ_SLOW=1 uv run pytest tests/test_e2e_smoke.py -q
```

Expected: 1 passed.

- [ ] **Step 3: Run the full suite**

```bash
uv run pytest -q
COLIRAZ_SLOW=1 uv run pytest -q
```

Expected: all green.

- [ ] **Step 4: Manual sanity — train on a real tiny dataset**

```bash
mkdir -p /tmp/coliraz-smoke
uv run python -c "
import cv2, numpy as np, os
rng = np.random.default_rng(0)
for i in range(32):
    cv2.imwrite(f'/tmp/coliraz-smoke/img{i}.png',
                rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
"
uv run coliraz scan-data --root /tmp/coliraz-smoke
uv run coliraz train --config configs/tiny.yaml --data /tmp/coliraz-smoke \
                     --total-steps 50 --run-name smoke
ls runs/smoke/samples/ runs/smoke/ckpt/
```

Expected: `runs/smoke/samples/latest.png` and `runs/smoke/ckpt/final.pt` both exist.

- [ ] **Step 5: Manual sanity — inference and export**

```bash
uv run coliraz infer --model runs/smoke/ckpt/final.pt \
                     --input /tmp/coliraz-smoke/img0.png \
                     --output /tmp/coliraz-smoke-out.png --input-size 64
uv run coliraz export --model runs/smoke/ckpt/final.pt \
                      --output /tmp/coliraz-smoke.onnx --input-size 64
ls -la /tmp/coliraz-smoke-out.png /tmp/coliraz-smoke.onnx
```

Expected: both files written; export prints parity-pass message.

- [ ] **Step 6: Commit the smoke test**

```bash
git add tests/test_e2e_smoke.py
git commit -m "$(cat <<'EOF'
coliraz: end-to-end smoke test (train -> ckpt -> preview -> infer)

Builds a synthetic 8-image dataset, runs 5 training steps with the
tiny model on CPU in fp32, asserts final checkpoint and preview PNG
are written, then loads the checkpoint and colorizes one image.
Guarded by COLIRAZ_SLOW=1 since it instantiates the full DDColor
model (a few seconds on CPU).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Report to the user**

Print a final summary including:
- `uv run coliraz train --config configs/tiny.yaml --data /path/to/images`
- where the preview PNG lands (`runs/<name>/samples/latest.png`)
- how to override losses (edit YAML or use a different `!preset`)
- how to switch to large (`--config configs/large.yaml`)
- how to enable torch.compile (`--compile`)

---

## Spec Coverage Check

| Spec section | Implemented by |
|---|---|
| 1 Goal | All tasks together |
| 2 Non-goals | Explicit (no DDP/Gradio/FID/W&B/video) |
| 3 Strategy (Approach A) | Task 7 (timm), Task 10 (SDPA), Task 8 (compile-safe spectral norm), Task 4 (Pydantic config) |
| 4 Project structure | Task 1 |
| 5 Model architecture | Tasks 7-11 |
| 6 Data pipeline | Tasks 5-6 |
| 7 Modular loss system | Tasks 13-18 |
| 8 Trainer | Task 23 |
| 9 Live UI | Task 22 |
| 10 Sample preview | Task 21 + Task 23 (background-thread render + atomic write + rotated history) |
| 11 Inference | Task 24 |
| 12 ONNX export | Task 25 |
| 13 CLI & config | Tasks 4, 26, 27 |
| 14 Testing | Every task ships with tests; Task 28 is the integration smoke |
| 15 Dependencies | Task 1 |
| 16 Migration steps | Task ordering (1→28) |
| 17 Open questions | None blocking |







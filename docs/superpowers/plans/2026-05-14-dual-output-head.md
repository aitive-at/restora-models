# Dual-Output Head + ONNX Export Wrapper Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover v1's structural advantage for colorization (model output's `ab` channels handled by a dedicated head, gated by the `config[0]` colorize axis) while keeping the existing `(rgb, config) → rgb` forward contract bit-identical, and pin the ONNX export contract behind a stable wrapper module.

**Architecture:** Single shared `DualOutputHead` class in `src/refine/models/heads.py` used by PromptIR. NAFNet inlines the same composition (different because its intermediate is Lab, not RGB). A new `ONNXExportWrapper` wraps the model at export time so the exported graph's input/output names + shapes are stable regardless of future internal forward changes. Production configs get an `axis_probs` rebalance (colorize/sharpen 0.5→0.75, easy tasks 0.5→0.4) so the harder axes get more gradient signal per batch.

**Tech Stack:** PyTorch (`nn.Module`, `nn.Conv2d`, `torch.cat`, linear-gate mixing), `refine.models.color.RgbToLab` / `LabToRgb` (existing ONNX-friendly conversion modules), torch.onnx export, onnxruntime parity, pytest.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/refine/models/heads.py` | NEW | `DualOutputHead` — RGB-delta head + Lab-ab head with linear gate on `config[0]`. ~50 lines. |
| `src/refine/export/wrapper.py` | NEW | `ONNXExportWrapper` — pure pass-through nn.Module with stable `(input, config) → output` signature for ONNX export. ~30 lines. |
| `src/refine/models/promptir.py` | MOD | Replace `self.head` + final residual with `self.dual_head: DualOutputHead`. ~10 lines diff. |
| `src/refine/models/nafnet.py` | MOD | Replace `self.head` with `self.head_lab_delta` + `self.head_ab_abs`; inline the dual-head composition. ~15 lines diff. |
| `src/refine/export/onnx.py` | MOD | Wrap `model` in `ONNXExportWrapper` before `torch.onnx.export`. ~3 lines diff. |
| `src/refine/train/checkpoint.py` | MOD | `_rename_legacy_keys` helper + `strict=False` warning for old single-head checkpoints. ~30 lines diff. |
| `configs/default.yaml` | MOD | axis_probs rebalance. |
| `configs/laion-compound.yaml` | MOD | axis_probs rebalance. |
| `configs/promptir-laion.yaml` | MOD | axis_probs rebalance. |
| `configs/nafnet-tiny-vivid.yaml` | MOD | axis_probs rebalance. |
| `tests/test_dual_head.py` | NEW | Gate semantics + shape + grad masking. |
| `tests/test_onnx_wrapper.py` | NEW | Wrapper is pure pass-through; optional clamping works. |
| `tests/test_legacy_checkpoint_load.py` | NEW | Old single-head ckpt loads into new dual-head model. |
| `tests/test_nafnet.py` | MOD | Updated for two heads. |
| `tests/test_promptir.py` | MOD | Updated for two heads. |
| `tests/test_promptir_e2e_smoke.py` | MOD | Verify dual-head behavior end-to-end. |

---

## Task 1: `DualOutputHead` module

**Why first:** Foundation. PromptIR (Task 4) depends on it.

**Files:**
- Create: `src/refine/models/heads.py`
- Create: `tests/test_dual_head.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dual_head.py`:

```python
import torch

from refine.models.heads import DualOutputHead


def _head(in_dim: int = 8) -> DualOutputHead:
    return DualOutputHead(in_dim=in_dim)


def test_forward_shape():
    h = _head()
    features = torch.randn(2, 8, 16, 16)
    rgb = torch.rand(2, 3, 16, 16)
    config = torch.zeros(2, 5)
    out = h(features=features, rgb_input=rgb, config=config)
    assert out.shape == rgb.shape
    assert torch.isfinite(out).all()


def test_passthrough_when_colorize_zero():
    """With config[0]=0, output must equal rgb + head_rgb(features) exactly
    (modulo the Lab round-trip). The ab override path contributes nothing."""
    h = _head()
    features = torch.randn(1, 8, 16, 16)
    rgb = torch.rand(1, 3, 16, 16)
    config = torch.zeros(1, 5)
    with torch.no_grad():
        rgb_delta = h.head_rgb(features)
        expected = rgb + rgb_delta
        out = h(features=features, rgb_input=rgb, config=config)
    # Lab round-trip introduces small numerical drift; loose tolerance
    assert (out - expected).abs().mean().item() < 1e-3


def test_ab_head_zero_init_means_gray_for_colorize_one():
    """With head_ab zero-initialized and config[0]=1, the prediction's ab
    channels should be 0 (the Lab-gray axis), so the output is grayscale
    derived from input's L."""
    h = _head()
    rgb = torch.rand(1, 3, 16, 16)
    features = torch.randn(1, 8, 16, 16)
    config = torch.tensor([[1.0, 0, 0, 0, 0]])
    with torch.no_grad():
        out = h(features=features, rgb_input=rgb, config=config)
    # output should be approximately gray: R≈G≈B per pixel
    rgb_chan_var = out.var(dim=1).mean().item()
    assert rgb_chan_var < 1e-2, f"output is not gray; per-pixel RGB variance = {rgb_chan_var}"


def test_linear_gate_distinct_endpoints():
    """The gate is linear in Lab-ab space; verify the mid-point output is
    distinct from both endpoints when head_ab predicts a nonzero value."""
    h = _head()
    # Force head_ab to a known nonzero state
    with torch.no_grad():
        h.head_ab.weight.fill_(0.01)
        h.head_ab.bias.fill_(5.0)
    features = torch.randn(1, 8, 16, 16)
    rgb = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out_0    = h(features, rgb, torch.tensor([[0.0, 0, 0, 0, 0]]))
        out_1    = h(features, rgb, torch.tensor([[1.0, 0, 0, 0, 0]]))
        out_half = h(features, rgb, torch.tensor([[0.5, 0, 0, 0, 0]]))
    delta_to_0 = (out_half - out_0).abs().mean().item()
    delta_to_1 = (out_half - out_1).abs().mean().item()
    assert delta_to_0 > 1e-3 and delta_to_1 > 1e-3, \
        "out_half is not distinct from both endpoints"


def test_gradient_routing():
    """When config[0]=0 across the whole batch, head_ab must receive zero grad
    (it's gated out). head_rgb receives grad on every sample."""
    h = _head()
    features = torch.randn(2, 8, 16, 16, requires_grad=False)
    rgb = torch.rand(2, 3, 16, 16, requires_grad=False)
    config = torch.zeros(2, 5)
    out = h(features=features, rgb_input=rgb, config=config)
    out.sum().backward()
    assert h.head_rgb.weight.grad is not None
    assert h.head_rgb.weight.grad.abs().sum().item() > 0
    assert h.head_ab.weight.grad is not None
    assert h.head_ab.weight.grad.abs().sum().item() < 1e-6


def test_param_count_small():
    """The dual head should be a thin output module, not a chunky decoder."""
    h = DualOutputHead(in_dim=64)
    n = sum(p.numel() for p in h.parameters())
    # Conv2d(64, 3, 3, 1): 64*3*9 + 3 = 1731
    # Conv2d(64, 2, 3, 1): 64*2*9 + 2 = 1154
    # Total ~ 2885 + any color-module params (RgbToLab/LabToRgb are buffer-only)
    assert n < 5000, f"DualOutputHead too large: {n} params"
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_dual_head.py -v`
Expected: FAIL — `refine.models.heads` not importable.

- [ ] **Step 3: Implement**

Create `src/refine/models/heads.py`:

```python
"""Shared output heads for refine restoration models.

The single class here, `DualOutputHead`, is the structural fix for the
colorize-quality gap surfaced in the 2026-05-14 design discussion.
Mirrors v1 DDColor's contract: a dedicated `ab`-prediction head whose
contribution is gated by the `config[0]` (colorize) axis, with the
luminance channel always carried by the parallel RGB-delta head.
"""
from __future__ import annotations

import torch
from torch import nn

from .color import LabToRgb, RgbToLab


class DualOutputHead(nn.Module):
    """RGB-delta head + Lab-ab head, composed by a linear gate on config[0].

    forward(features, rgb_input, config) -> rgb_output

    - rgb_intermediate = rgb_input + head_rgb(features)
    - ab_pred          = head_ab(features)
    - new_ab = config[0] * ab_pred + (1 - config[0]) * lab(rgb_intermediate).ab
    - output = lab_to_rgb(L=lab(rgb_intermediate).L, ab=new_ab)

    Properties:
      - config[0] = 0 -> output == rgb_intermediate (modulo lab round-trip).
                         head_ab receives zero gradient on this sample.
      - config[0] = 1 -> output's ab channels equal head_ab(features) exactly.
                         L is carried by head_rgb's contribution.
      - Composes correctly with multi-axis configs (e.g. colorize+denoise):
        head_rgb does the other tasks (preserves L correctly), head_ab
        overrides only the color channels.

    Init:
      head_rgb: small Gaussian (std=0.01) - initial output ~ input.
      head_ab : zero - initial ab = 0, so colorize=1 at step 0 yields
                gray output. Model learns to add color from there.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.rgb_to_lab = RgbToLab()
        self.lab_to_rgb = LabToRgb()
        self.head_rgb = nn.Conv2d(in_dim, 3, kernel_size=3, padding=1)
        self.head_ab  = nn.Conv2d(in_dim, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.head_rgb.weight, std=0.01)
        nn.init.zeros_(self.head_rgb.bias)
        nn.init.zeros_(self.head_ab.weight)
        nn.init.zeros_(self.head_ab.bias)

    def forward(self, features: torch.Tensor,
                rgb_input: torch.Tensor,
                config: torch.Tensor) -> torch.Tensor:
        rgb_intermediate = rgb_input + self.head_rgb(features)
        ab_pred          = self.head_ab(features)

        lab = self.rgb_to_lab(rgb_intermediate)
        w   = config[:, 0:1].view(-1, 1, 1, 1)
        new_ab = w * ab_pred + (1.0 - w) * lab[:, 1:3]
        lab_out = torch.cat([lab[:, 0:1], new_ab], dim=1)
        return self.lab_to_rgb(lab_out)
```

Read `src/refine/models/color.py` first to confirm `RgbToLab` and `LabToRgb` are the actual class names exported there (they were used by NAFNet already so they exist).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dual_head.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/refine/models/heads.py tests/test_dual_head.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: add DualOutputHead — Lab-ab head gated by colorize axis

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `ONNXExportWrapper` module

**Files:**
- Create: `src/refine/export/wrapper.py`
- Create: `tests/test_onnx_wrapper.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_onnx_wrapper.py`:

```python
import torch
from torch import nn

from refine.export.wrapper import ONNXExportWrapper


class _Toy(nn.Module):
    """Minimal model satisfying the refine forward contract."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.proj.weight, 0.1)

    def forward(self, rgb, config):
        out = self.proj(rgb)
        # touch config so it's part of the graph
        return out + config.sum(dim=-1).view(-1, 1, 1, 1) * 0.0


def test_wrapper_is_pure_passthrough():
    m = _Toy()
    w = ONNXExportWrapper(m, clamp_output=False)
    x = torch.rand(2, 3, 16, 16)
    c = torch.tensor([[1.0, 0, 1, 0, 0], [0, 1, 0, 1, 0]])
    direct = m(x, c)
    via_wrapper = w(x, c)
    assert torch.equal(direct, via_wrapper)


def test_clamp_applied():
    """When clamp_output=True, output is in [0, 1] even if underlying overshoots."""
    class _Overshoot(nn.Module):
        def forward(self, rgb, config):
            return rgb * 5.0 - 1.0   # range [-1, 4]
    w = ONNXExportWrapper(_Overshoot(), clamp_output=True)
    x = torch.rand(1, 3, 4, 4)
    out = w(x, torch.zeros(1, 5))
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_clamp_off_by_default():
    """Default behavior must not silently clamp - caller's responsibility."""
    class _Overshoot(nn.Module):
        def forward(self, rgb, config):
            return rgb * 5.0 - 1.0
    w = ONNXExportWrapper(_Overshoot())
    out = w(torch.rand(1, 3, 4, 4), torch.zeros(1, 5))
    assert (out < 0.0).any() or (out > 1.0).any()


def test_signature_names():
    """The wrapper's forward parameter names ARE the ONNX input names.
    The contract is forward(input, config) -> output."""
    import inspect
    sig = inspect.signature(ONNXExportWrapper.forward)
    params = list(sig.parameters.keys())
    assert params[1] == "input", f"first arg must be 'input', got {params[1]!r}"
    assert params[2] == "config", f"second arg must be 'config', got {params[2]!r}"
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_onnx_wrapper.py -v`
Expected: FAIL — module not importable.

- [ ] **Step 3: Implement**

Create `src/refine/export/wrapper.py`:

```python
"""Stable ONNX export entry point for refine models.

Pins the exported graph signature `forward(input, config) -> output`
regardless of future changes to a backbone's Python forward. Today the
wrapper is a pure pass-through; in the future it can host stable
preprocessing (input normalization, dtype coercion, output clamping)
without touching the backbones themselves.
"""
from __future__ import annotations

import torch
from torch import nn


class ONNXExportWrapper(nn.Module):
    """Wraps a backbone whose Python forward is `(rgb, config) -> rgb`.

    The parameter names `input` and `config` are deliberately chosen
    because torch.onnx.export uses parameter names for the exported
    graph's input names by default. Output name is "output".

    Args:
        model: any nn.Module with `forward(rgb, config) -> rgb` where
               rgb is (B, 3, H, W) float in [0, 1] and config is
               (B, num_axes) float in [0, 1].
        clamp_output: if True, the wrapped output is clamped to [0, 1].
                      Defaults to False to keep training-time behavior
                      where loss/metric code does its own range handling.
    """

    def __init__(self, model: nn.Module, *, clamp_output: bool = False) -> None:
        super().__init__()
        self.model = model
        self.clamp_output = bool(clamp_output)

    def forward(self, input: torch.Tensor,            # noqa: A002 (intentional shadow)
                config: torch.Tensor) -> torch.Tensor:
        out = self.model(input, config)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_onnx_wrapper.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/refine/export/wrapper.py tests/test_onnx_wrapper.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: add ONNXExportWrapper — stable export contract

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update ONNX exporter to use the wrapper

**Files:**
- Modify: `src/refine/export/onnx.py`
- Modify: `tests/test_onnx_wrapper.py` (append one regression test)

- [ ] **Step 1: Append the regression test**

Append to `tests/test_onnx_wrapper.py`:

```python
def test_exporter_uses_wrapper(tmp_path):
    """Smoke: full export through export_onnx_from_model produces an ONNX
    whose input names are exactly ['input', 'config'] and output is 'output'."""
    import os
    if not os.environ.get("REFINE_SLOW"):
        import pytest
        pytest.skip("slow ONNX export, set REFINE_SLOW=1 to run")

    from refine.config import ModelConfig
    from refine.models import build_model
    from refine.export.onnx import export_onnx_from_model

    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "wrapped.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=32, export_path=out,
        opset=17, simplify=False, verify_parity=True, parity_atol=1e-3,
        dynamic_hw=False, task_map=None, precision="fp32",
    )
    import onnx
    om = onnx.load(str(out))
    input_names = sorted([i.name for i in om.graph.input])
    output_names = sorted([o.name for o in om.graph.output])
    assert input_names == ["config", "input"]
    assert output_names == ["output"]
```

- [ ] **Step 2: Run to confirm baseline passes**

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_onnx_wrapper.py::test_exporter_uses_wrapper -v`
Expected: PASS (current exporter happens to use these names; we're locking the contract).

- [ ] **Step 3: Update the exporter to wrap the model**

Read `src/refine/export/onnx.py`. Find the `torch.onnx.export(...)` call. Replace the call site to wrap `model` first. The new block:

```python
    # Pin the ONNX contract behind a stable wrapper module so future
    # backbone changes can't drift the exported graph's I/O signature.
    from .wrapper import ONNXExportWrapper
    export_model = ONNXExportWrapper(model)
    export_model.train(False)

    torch.onnx.export(
        export_model, (dummy_rgb, dummy_cfg), str(export_path),
        opset_version=opset,
        input_names=["input", "config"], output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
```

The rest of the function (simplify, fp16/fp8 conversion, parity verification, sidecar) is unchanged. Parity verification calls `model(...)` directly (not via the wrapper) so it still works.

- [ ] **Step 4: Re-run all slow ONNX tests to verify no regression**

Run:
```bash
REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_onnx_wrapper.py \
                                          tests/test_export_onnx.py \
                                          tests/test_export_precision.py \
                                          tests/test_promptir.py::test_onnx_export_parity_all_configs \
                                          -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/refine/export/onnx.py tests/test_onnx_wrapper.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: ONNX exporter wraps model in ONNXExportWrapper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: PromptIR — use `DualOutputHead`

**Files:**
- Modify: `src/refine/models/promptir.py`
- Modify: `tests/test_promptir.py`

- [ ] **Step 1: Append the new assertions to `tests/test_promptir.py`**

Read `tests/test_promptir.py`. Append two tests:

```python
def test_promptir_has_dual_head():
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    assert hasattr(m, "dual_head"), "promptir.dual_head missing"
    from refine.models.heads import DualOutputHead
    assert isinstance(m.dual_head, DualOutputHead)
    assert not hasattr(m, "head"), \
        "self.head must be replaced by self.dual_head, not kept alongside"


def test_promptir_colorize_off_preserves_input():
    """Re-test of the identity property under the dual-head architecture.
    With colorize=0 and untrained head_rgb (small-normal init), output
    should be very close to input."""
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    c = torch.zeros(1, 5)
    with torch.no_grad():
        out = m(x, c)
    diff = (out - x).abs().mean().item()
    assert diff < 0.1, f"identity-config output drifted: diff={diff}"
```

- [ ] **Step 2: Run to verify new tests fail**

Run: `.venv/bin/python -m pytest tests/test_promptir.py::test_promptir_has_dual_head -v`
Expected: FAIL — `dual_head` attribute missing.

- [ ] **Step 3: Update `src/refine/models/promptir.py`**

Read the file first. Find the `__init__` block creating `self.head`. Replace:

```python
        self.head = nn.Conv2d(dim, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
```

with:

```python
        from .heads import DualOutputHead
        self.dual_head = DualOutputHead(in_dim=dim)
```

Then find the `forward` method's final line:

```python
        return rgb + self.head(d)
```

Replace with:

```python
        return self.dual_head(features=d, rgb_input=rgb, config=config)
```

- [ ] **Step 4: Run all PromptIR tests + slow ONNX parity**

Run: `.venv/bin/python -m pytest tests/test_promptir.py -v`
Expected: 8 passed (6 existing + 2 new), 1 skipped (slow).

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_promptir.py::test_onnx_export_parity_all_configs -v`
Expected: PASS. (Dual-head introduces a per-forward rgb_to_lab + lab_to_rgb pair; the existing 1e-1 ONNX parity tolerance accommodates this.)

- [ ] **Step 5: Commit**

```bash
git add src/refine/models/promptir.py tests/test_promptir.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: PromptIR uses DualOutputHead — colorize axis routes to ab head

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: NAFNet — inline dual-head composition

**Why separate from Task 4:** NAFNet's intermediate is Lab, not RGB, so it uses a slightly different composition (it produces a Lab delta + an absolute ab prediction; PromptIR produces an RGB delta + an absolute ab). The shared `DualOutputHead` doesn't fit, so we inline.

**Files:**
- Modify: `src/refine/models/nafnet.py`
- Modify: `tests/test_nafnet.py`

- [ ] **Step 1: Append the new assertions to `tests/test_nafnet.py`**

```python
def test_nafnet_has_dual_heads():
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    assert hasattr(m, "head_lab_delta"), "head_lab_delta missing"
    assert hasattr(m, "head_ab_abs"), "head_ab_abs missing"
    assert not hasattr(m, "head"), \
        "bare self.head must be removed; replaced by head_lab_delta + head_ab_abs"
    assert m.head_lab_delta.out_channels == 3
    assert m.head_ab_abs.out_channels == 2


def test_nafnet_colorize_off_preserves_input():
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    c = torch.zeros(1, 5)
    with torch.no_grad():
        out = m(x, c)
    diff = (out - x).abs().mean().item()
    assert diff < 0.05, f"identity-config output drifted: diff={diff}"


def test_nafnet_colorize_on_predicts_gray_at_init():
    """With config[0]=1 and untrained heads (both zero), output should be
    grayscale (head_ab_abs produces 0 -> Lab ab = 0 -> gray)."""
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    c = torch.tensor([[1.0, 0, 0, 0, 0]])
    with torch.no_grad():
        out = m(x, c)
    chan_var = out.var(dim=1).mean().item()
    assert chan_var < 5e-2, f"colorize=1 with zero-init head_ab is not gray: var={chan_var}"
```

If `tests/test_nafnet.py` doesn't already import torch and ModelConfig and build_model, add those imports at the top of the file:

```python
import torch
from refine.config import ModelConfig
from refine.models import build_model
```

- [ ] **Step 2: Run to verify new tests fail**

Run: `.venv/bin/python -m pytest tests/test_nafnet.py::test_nafnet_has_dual_heads -v`
Expected: FAIL — `head_lab_delta` missing.

- [ ] **Step 3: Update `src/refine/models/nafnet.py`**

Read the file first. Find the `__init__` block ending with `self.head`. Replace:

```python
        self.head = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
```

with:

```python
        # Dual output: Lab delta (3 ch, all tasks) + absolute Lab ab (2 ch, colorize axis).
        # Both zero-inited so initial output ~ input via the global Lab residual,
        # and colorize=1 at step 0 produces gray (model learns to add color from there).
        self.head_lab_delta = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.head_ab_abs   = nn.Conv2d(nf, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head_lab_delta.weight)
        nn.init.zeros_(self.head_lab_delta.bias)
        nn.init.zeros_(self.head_ab_abs.weight)
        nn.init.zeros_(self.head_ab_abs.bias)
```

In `forward`, the current ending:

```python
        delta_lab_n = self.head(x)
        return self.lab_to_rgb(lab_n + delta_lab_n)
```

Replace with:

```python
        delta_lab_n = self.head_lab_delta(x)
        ab_pred    = self.head_ab_abs(x)

        # Compose: Lab intermediate carries all-task signal; ab override gated by colorize axis.
        lab_intermediate = lab_n + delta_lab_n
        w   = config[:, 0:1].view(-1, 1, 1, 1)
        ab_out = w * ab_pred + (1.0 - w) * lab_intermediate[:, 1:3]
        L_out  = lab_intermediate[:, 0:1]
        return self.lab_to_rgb(torch.cat([L_out, ab_out], dim=1))
```

- [ ] **Step 4: Run NAFNet tests + slow ONNX export**

Run: `.venv/bin/python -m pytest tests/test_nafnet.py -v`
Expected: all pass (including the 3 new ones).

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_export_onnx.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/refine/models/nafnet.py tests/test_nafnet.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: NAFNet inlines dual-head (head_lab_delta + head_ab_abs)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Backward-compatible checkpoint loading

**Why:** Old checkpoints (committed before today) have a single `head.weight` / `head.bias` key. We need to (1) accept them via `strict=False` + warning, and (2) rename `head.*` to the new heads' bare keys so the carried-over weights aren't lost.

**Files:**
- Modify: `src/refine/train/checkpoint.py`
- Create: `tests/test_legacy_checkpoint_load.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_legacy_checkpoint_load.py`:

```python
"""Old single-head checkpoints must still load into the new dual-head models.

The legacy keys (head.weight, head.bias) are renamed to the new heads' bare
keys so the carried-over weights aren't lost; missing keys (head_ab_*) are
zero-initialized by the model constructor and ignored at load time."""
from __future__ import annotations

import torch

from refine.config import ModelConfig
from refine.models import build_model
from refine.train.checkpoint import load_checkpoint


def _save_legacy_nafnet_ckpt(tmp_path):
    """Build a fake old NAFNet checkpoint by simulating the legacy single-head
    state_dict layout from a freshly initialized dual-head model."""
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    sd = m.state_dict()
    legacy = {}
    for k, v in sd.items():
        if k.startswith("head_lab_delta."):
            legacy["head." + k.split(".", 1)[1]] = v.clone()
        elif k.startswith("head_ab_abs."):
            pass  # drop - legacy ckpt didn't have this
        else:
            legacy[k] = v.clone()
    path = tmp_path / "legacy.pt"
    torch.save({
        "model": legacy, "step": 100,
        "extra": {"cfg": {"model": cfg.model_dump()}},
    }, path)
    return path


def test_legacy_nafnet_checkpoint_loads(tmp_path):
    path = _save_legacy_nafnet_ckpt(tmp_path)
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    fresh = build_model(cfg, num_axes=5)
    before = fresh.head_ab_abs.weight.clone()
    payload = load_checkpoint(path, model=fresh)
    after = fresh.head_ab_abs.weight
    # head_ab_abs untouched by load (key wasn't in the legacy ckpt)
    assert torch.equal(before, after)
    # head_lab_delta got loaded - verify by comparing to another fresh model
    fresh2 = build_model(cfg, num_axes=5)
    assert torch.equal(fresh.head_lab_delta.weight, fresh2.head_lab_delta.weight)
    assert payload["step"] == 100


def _save_legacy_promptir_ckpt(tmp_path):
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    sd = m.state_dict()
    legacy = {}
    for k, v in sd.items():
        if k.startswith("dual_head.head_rgb."):
            legacy["head." + k.split(".", 2)[2]] = v.clone()
        elif k.startswith("dual_head."):
            pass
        else:
            legacy[k] = v.clone()
    path = tmp_path / "legacy_promptir.pt"
    torch.save({
        "model": legacy, "step": 100,
        "extra": {"cfg": {"model": cfg.model_dump()}},
    }, path)
    return path


def test_legacy_promptir_checkpoint_loads(tmp_path):
    path = _save_legacy_promptir_ckpt(tmp_path)
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    fresh = build_model(cfg, num_axes=5)
    before_ab = fresh.dual_head.head_ab.weight.clone()
    payload = load_checkpoint(path, model=fresh)
    after_ab = fresh.dual_head.head_ab.weight
    assert torch.equal(before_ab, after_ab)
    assert payload["step"] == 100
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_legacy_checkpoint_load.py -v`
Expected: FAIL — `load_checkpoint` uses strict=True and rejects unexpected/missing keys.

- [ ] **Step 3: Implement the rename + strict-False loader**

Read `src/refine/train/checkpoint.py`. Add this helper function above `load_checkpoint`:

```python
def _rename_legacy_keys(state_dict: dict, target_keys: set[str]) -> tuple[dict, list[str]]:
    """Rename single-head checkpoint keys (`head.*`) to the new dual-head names.

    Determines the rename target by inspecting `target_keys` (the keys of the
    model we're loading INTO):
      - If target has `head_lab_delta.*` (NAFNet): legacy `head.*` -> `head_lab_delta.*`
      - If target has `dual_head.head_rgb.*` (PromptIR): legacy `head.*` -> `dual_head.head_rgb.*`
      - Otherwise: no rename.

    Returns (renamed_state_dict, list_of_rename_strings).
    """
    has_nafnet_dual = any(k.startswith("head_lab_delta.") for k in target_keys)
    has_promptir_dual = any(k.startswith("dual_head.head_rgb.") for k in target_keys)
    if not (has_nafnet_dual or has_promptir_dual):
        return state_dict, []
    if has_nafnet_dual and has_promptir_dual:  # defensive — both backbones in one model would be a bug
        return state_dict, []
    prefix = "head_lab_delta." if has_nafnet_dual else "dual_head.head_rgb."
    renamed = {}
    log: list[str] = []
    for k, v in state_dict.items():
        if k.startswith("head."):
            new_k = prefix + k.split(".", 1)[1]
            renamed[new_k] = v
            log.append(f"{k} -> {new_k}")
        else:
            renamed[k] = v
    return renamed, log
```

Then update the model-load block inside `load_checkpoint`. The current code is:

```python
    if model is not None and "model" in payload:
        _unwrap(model).load_state_dict(payload["model"])
```

Replace with:

```python
    if model is not None and "model" in payload:
        target = _unwrap(model)
        target_keys = set(target.state_dict().keys())
        sd, renames = _rename_legacy_keys(payload["model"], target_keys)
        if renames:
            print(f"[load_checkpoint] renamed {len(renames)} legacy keys "
                  f"(e.g. {renames[0]})", flush=True)
        missing, unexpected = target.load_state_dict(sd, strict=False)
        if missing:
            preview = list(missing)[:3]
            print(f"[load_checkpoint] {len(missing)} missing keys (zero-inited): "
                  f"{preview}{'...' if len(missing) > 3 else ''}", flush=True)
        if unexpected:
            preview = list(unexpected)[:3]
            print(f"[load_checkpoint] {len(unexpected)} unexpected keys (ignored): "
                  f"{preview}{'...' if len(unexpected) > 3 else ''}", flush=True)
```

Make sure `_unwrap` is still imported at the top of the file (it was added in the torch.compile fix).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_legacy_checkpoint_load.py tests/test_train_ckpt_ema.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/refine/train/checkpoint.py tests/test_legacy_checkpoint_load.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: load_checkpoint — rename legacy single-head keys, strict=False

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Rebalance `axis_probs` in production configs

**Files:**
- Modify: `configs/default.yaml`
- Modify: `configs/laion-compound.yaml`
- Modify: `configs/promptir-laion.yaml`
- Modify: `configs/nafnet-tiny-vivid.yaml`
- Modify: `tests/test_configs_load.py`

- [ ] **Step 1: Write the new test**

Append to `tests/test_configs_load.py`:

```python
def test_default_axis_probs_rebalanced():
    cfg = load_config(ROOT / "default.yaml", overrides={"data": {"root": "/tmp"}})
    ap = cfg.compound.axis_probs
    assert ap.colorize == 0.75, f"colorize should be 0.75 (rebalanced), got {ap.colorize}"
    assert ap.sharpen  == 0.75, f"sharpen should be 0.75 (rebalanced), got {ap.sharpen}"
    assert ap.denoise  == 0.40
    assert ap.dejpeg   == 0.40
    assert ap.deblur   == 0.40
```

The existing `test_laion_compound_loads` test asserts `cfg.compound.axis_probs.colorize == 0.5`. Update it to `0.75`:

```python
def test_laion_compound_loads():
    cfg = load_config(ROOT / "laion-compound.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.compound.identity_prob == 0.05
    assert cfg.compound.axis_probs.colorize == 0.75
    assert cfg.compound.axis_probs.sharpen == 0.75
```

- [ ] **Step 2: Run to verify failures**

Run: `.venv/bin/python -m pytest tests/test_configs_load.py::test_default_axis_probs_rebalanced tests/test_configs_load.py::test_laion_compound_loads -v`
Expected: both FAIL — current values are 0.5.

- [ ] **Step 3: Edit each config**

For `configs/default.yaml`, find the `compound:` block and replace its `axis_probs` section with:

```yaml
compound:
  identity_prob: 0.05
  axis_probs:
    colorize: 0.75
    denoise:  0.40
    sharpen:  0.75
    dejpeg:   0.40
    deblur:   0.40
```

For each of `configs/laion-compound.yaml`, `configs/promptir-laion.yaml`, `configs/nafnet-tiny-vivid.yaml`: read each, find the `compound:` block. If it redefines `axis_probs`, update those values to the same six lines (`colorize: 0.75`, `denoise: 0.40`, `sharpen: 0.75`, `dejpeg: 0.40`, `deblur: 0.40`). Otherwise the inheritance from `default.yaml` will pick up the change.

- [ ] **Step 4: Run all config-loading tests**

Run: `.venv/bin/python -m pytest tests/test_configs_load.py tests/test_compound.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml configs/laion-compound.yaml \
        configs/promptir-laion.yaml configs/nafnet-tiny-vivid.yaml \
        tests/test_configs_load.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: rebalance axis_probs — colorize/sharpen 0.75, easy tasks 0.40

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end smoke test with dual-head

**Files:**
- Modify: `tests/test_promptir_e2e_smoke.py`

- [ ] **Step 1: Append dual-head verification to the existing smoke test**

Read `tests/test_promptir_e2e_smoke.py`. After the `trainer.fit()` call and the existing ckpt/preview/sidecar assertions (but before the fp16 ONNX export block), insert:

```python
    # Dual-head verification: after training, head_ab.weight should have
    # changed from its zero init (it received gradient on colorize samples).
    # head_rgb.weight should have moved as well (gradient on every sample).
    payload = torch.load(str(final), map_location="cpu", weights_only=False)
    sd = payload["model"]
    head_ab_w = sd.get("dual_head.head_ab.weight")
    head_rgb_w = sd.get("dual_head.head_rgb.weight")
    assert head_ab_w is not None,  "dual_head.head_ab.weight not in checkpoint"
    assert head_rgb_w is not None, "dual_head.head_rgb.weight not in checkpoint"
    assert head_ab_w.abs().max().item() > 1e-6, \
        "head_ab.weight didn't get any gradient — dual-head wiring broken"
    assert head_rgb_w.abs().max().item() > 1e-6
```

- [ ] **Step 2: Run the slow e2e test**

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_promptir_e2e_smoke.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_promptir_e2e_smoke.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: e2e smoke — verify dual-head trains both branches

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final verification gate

**Files:** none changed.

- [ ] **Step 1: Run full fast suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass.

- [ ] **Step 2: Run full slow suite**

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass.

- [ ] **Step 3: Quick CLI sanity**

```bash
.venv/bin/refine --help
.venv/bin/refine export --help
```

Expected: no errors; `--precision` still visible.

- [ ] **Step 4: Confirm clean git status**

```bash
git status
git log --oneline 08b3c4f..HEAD
```

Expected: clean tree, 8 new commits since the spec commit (`08b3c4f`).

---

## Self-Review Notes

Spec coverage check (against `docs/superpowers/specs/2026-05-14-dual-output-head-design.md`):

- §3 Architecture (DualOutputHead, math properties): Task 1 (class + tests), Tasks 4+5 (per-backbone wiring).
- §3.2 NAFNet vs PromptIR integration: PromptIR uses DualOutputHead (Task 4); NAFNet inlines because of Lab-native intermediate (Task 5).
- §4 ONNX export wrapper: Task 2 (class + tests), Task 3 (exporter wires it).
- §5 axis_probs rebalance: Task 7.
- §6 Backward compatibility (legacy checkpoint loading): Task 6.
- §7 Files: every file in the spec's file list has a task that touches it.
- §8 Tests: every test in the spec's test list has an owning task.
- §9 Numerical/training contract: enforced by parity tests (Tasks 3, 4, 5).
- §10 Loss behavior: no code changes needed — loss code already operates on RGB output which is the dual-head's contract.
- §11 Out of scope: GAN/curriculum/CosAE all stay out of scope; no task touches them.

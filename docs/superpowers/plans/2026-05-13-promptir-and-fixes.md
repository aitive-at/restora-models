# PromptIR + Colorization Fix + SR Preview + ONNX Precision Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver four changes in one branch: (1) fix infrared-look colorization by adding a Lab ab-channel chroma loss and rebalancing loss presets, (2) split the `sharpen` preview into per-factor rows (sr2x/sr4x/sr8x), (3) implement the PromptIR backbone per the 2026-05-13 design spec, (4) extend ONNX export with `--precision fp16/fp8/fp4` (fp4 stubbed with a helpful error).

**Architecture:** All changes are additive — they coexist with the current `nafnet` backbone and existing loss/preview/export code paths. PromptIR is a second entry in `MODEL_REGISTRY`; chroma loss is a new entry in `LOSS_REGISTRY`; SR preview adds rows by forcing single-element `factor_choices` on `SharpenDegradation`; precision flag is a parameter on the existing exporter. No checkpoint or config format changes break existing runs.

**Tech Stack:** PyTorch (Conv2d, MultiheadAttention, PixelUnshuffle/Shuffle, F.softmax, einsum, F.interpolate), torch.onnx export, onnxruntime, onnxconverter-common (for fp16 conversion), pytest, Typer (CLI).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/refine/losses/chroma.py` | NEW | `chroma_lab` loss — Lab ab-channel L1 between pred & clean |
| `src/refine/losses/__init__.py` | MOD | Import `chroma` so registry sees it |
| `src/refine/config.py` | MOD | (a) Add chroma_lab to `standard` + `vivid` presets; (b) add 3 PromptIR override fields |
| `src/refine/train/trainer.py` | MOD | Replace single `sharpen-only` preview row with `sharpen-2x/4x/8x` rows; force factor via per-row `SharpenDegradation` instances |
| `src/refine/models/restormer_block.py` | NEW | `TransformerBlock` with MDTA + GDFN + AdaLN modulation (single class) |
| `src/refine/models/prompt_block.py` | NEW | `PromptBlock`: config-driven mix of N learnable prompts |
| `src/refine/models/promptir.py` | NEW | `@register_model("promptir")` backbone (U-Net with restormer blocks + prompt injection) |
| `src/refine/models/__init__.py` | MOD | Import `promptir` so registry sees it |
| `src/refine/export/onnx.py` | MOD | Accept `precision: Literal["fp32","fp16","fp8","fp4"]`; convert post-export for fp16; stub fp8/fp4 |
| `src/refine/cli.py` | MOD | Add `--precision` option on `export` command |
| `configs/promptir-laion.yaml` | NEW | Production-size PromptIR-large training config for laion-images |
| `tests/test_chroma_loss.py` | NEW | shape + zero-on-identical + non-zero-on-different |
| `tests/test_preview_sr_factors.py` | NEW | preview includes `sharpen-2x/4x/8x` rows; degraded image differs across factors |
| `tests/test_restormer_block.py` | NEW | forward shape + backward + AdaLN actually changes output |
| `tests/test_prompt_block.py` | NEW | config-determinism + shape + grad |
| `tests/test_promptir.py` | NEW | forward shape + param count + identity-config + ONNX parity (slow) |
| `tests/test_export_precision.py` | NEW | fp16 export round-trip + parity; fp8/fp4 raise appropriate errors |

---

## Task 1: chroma_lab loss (Lab ab-channel L1)

**Why this task first:** This is the smallest, highest-leverage change for the colorization issue. Current "infrared map" output is caused by the unopposed colorfulness loss; introducing a Lab-space chroma anchor lets the loss preset rebalance in Task 2.

**Files:**
- Create: `src/refine/losses/chroma.py`
- Create: `tests/test_chroma_loss.py`
- Modify: `src/refine/losses/__init__.py` (one import line)

- [ ] **Step 1: Write the failing test**

Create `tests/test_chroma_loss.py`:

```python
import torch

from restora_models.losses.registry import LOSS_REGISTRY, LossContext, build_loss


def _ctx(pred, clean):
    return LossContext(
        pred_rgb=pred, clean_rgb=clean,
        degraded_rgb=torch.zeros_like(pred), config=torch.zeros(pred.shape[0], 5),
        axes_active=["colorize"] * pred.shape[0],
    )


def test_chroma_loss_registered():
    assert "chroma_lab" in LOSS_REGISTRY


def test_chroma_loss_zero_on_identical():
    rgb = torch.rand(2, 3, 32, 32)
    loss = build_loss("chroma_lab")
    out = loss(_ctx(rgb.clone(), rgb.clone()))
    assert out.item() == 0.0 or out.item() < 1e-5


def test_chroma_loss_positive_on_different_hue():
    # red vs green of same luminance — same L, different ab
    red   = torch.zeros(1, 3, 16, 16); red[:, 0] = 1.0
    green = torch.zeros(1, 3, 16, 16); green[:, 1] = 1.0
    loss = build_loss("chroma_lab")
    out = loss(_ctx(red, green))
    assert out.item() > 1.0   # ab channels span ~[-128, 127]; opposite hues mean tens of units


def test_chroma_loss_ignores_luminance():
    # same hue, different brightness → ab roughly equal → small loss
    bright = torch.full((1, 3, 16, 16), 0.8); bright[:, 0] = 1.0
    dark   = torch.full((1, 3, 16, 16), 0.2); dark[:, 0] = 0.4
    loss = build_loss("chroma_lab")
    bright_dark = loss(_ctx(bright, dark)).item()

    # vs hue-flipped (red↔green) at same brightness — should be much larger
    red    = torch.zeros(1, 3, 16, 16); red[:, 0]   = 0.8
    green  = torch.zeros(1, 3, 16, 16); green[:, 1] = 0.8
    hue_flip = loss(_ctx(red, green)).item()

    assert hue_flip > bright_dark * 2


def test_chroma_loss_backprop():
    pred  = torch.rand(1, 3, 16, 16, requires_grad=True)
    clean = torch.rand(1, 3, 16, 16)
    loss = build_loss("chroma_lab")
    out = loss(_ctx(pred, clean))
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chroma_loss.py -v`
Expected: FAIL — `chroma_lab` not in `LOSS_REGISTRY` (`KeyError` from `build_loss`).

- [ ] **Step 3: Write minimal implementation**

Create `src/refine/losses/chroma.py`:

```python
"""Chroma loss — L1 on Lab ab-channels.

Anchors hue+saturation to ground truth, independent of luminance.
Used to counter the colorfulness-loss-driven "infrared map" failure
mode where the model maximizes opponent-color variance without
respecting the true hue.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from restora_models.utils.color import rgb_to_lab

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("chroma_lab")
class ChromaLabLoss(RestorationLoss):
    """L1 between predicted and ground-truth `ab` channels in CIELab.

    Inputs are RGB [0, 1] sRGB; converted on the fly. The `ab` channels
    have approximate range [-128, 127], so the raw loss is in the same
    units — at a weight of 1.0 it's commensurate with l1_rgb at weight
    ~0.01. Callers should weight accordingly (the loss preset does).
    """

    def __init__(self, scale: float = 0.01) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        pred_lab  = rgb_to_lab(ctx.pred_rgb.clamp(0, 1))
        clean_lab = rgb_to_lab(ctx.clean_rgb.clamp(0, 1))
        return F.l1_loss(pred_lab[:, 1:3], clean_lab[:, 1:3]) * self.scale
```

Modify `src/refine/losses/__init__.py` — add this import line near the other loss imports (preserve existing imports):

```python
from . import chroma as _chroma  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_chroma_loss.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/refine/losses/chroma.py tests/test_chroma_loss.py src/refine/losses/__init__.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: add chroma_lab loss (Lab ab-channel L1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rebalance loss presets to use chroma_lab on colorize

**Why:** The unopposed `colorfulness` loss maximizes opponent-color variance. Adding `chroma_lab` anchors hue to GT; reducing colorfulness weight to a weak prior fixes the infrared look without losing the "make output vivid" intent.

**Files:**
- Modify: `src/refine/config.py` (the `_LOSS_PRESETS` dict)
- Create: `tests/test_loss_presets.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_loss_presets.py`:

```python
def test_standard_preset_includes_chroma_lab():
    from restora_models.config import expand_loss_preset
    losses = expand_loss_preset("standard")
    names = [l.name for l in losses]
    assert "chroma_lab" in names
    # colorfulness weight reduced or removed
    cf = [l for l in losses if l.name == "colorfulness"]
    if cf:
        assert cf[0].weight <= 0.05, f"colorfulness weight too high: {cf[0].weight}"


def test_vivid_preset_keeps_chroma_anchor():
    from restora_models.config import expand_loss_preset
    losses = expand_loss_preset("vivid")
    names = [l.name for l in losses]
    assert "chroma_lab" in names


def test_full_preset_includes_chroma_lab():
    from restora_models.config import expand_loss_preset
    losses = expand_loss_preset("full")
    names = [l.name for l in losses]
    assert "chroma_lab" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loss_presets.py -v`
Expected: FAIL — `chroma_lab` not in `standard` preset.

- [ ] **Step 3: Update the loss presets**

Edit `src/refine/config.py`. Find the `_LOSS_PRESETS` dict (search for `"standard": [`) and replace the three entries `standard`, `vivid`, `full` with:

```python
_LOSS_PRESETS: dict[str, list[dict]] = {
    "standard": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "chroma_lab", "weight": 1.0, "apply_to_axes": ["colorize"]},
        {"name": "colorfulness", "weight": 0.02, "apply_to_axes": ["colorize"]},
        {"name": "freq_l1", "weight": 0.2, "apply_to_axes": ["sharpen", "deblur"]},
    ],
    "vivid": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "chroma_lab", "weight": 1.0, "apply_to_axes": ["colorize"]},
        {"name": "colorfulness", "weight": 0.05, "apply_to_axes": ["colorize"]},
        {"name": "freq_l1", "weight": 0.2, "apply_to_axes": ["sharpen", "deblur"]},
    ],
    "full": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "chroma_lab", "weight": 1.0, "apply_to_axes": ["colorize"]},
        {"name": "colorfulness", "weight": 0.02, "apply_to_axes": ["colorize"]},
        {"name": "freq_l1", "weight": 0.2, "apply_to_axes": ["sharpen", "deblur"]},
        {"name": "gan", "weight": 0.1, "config": {"gan_type": "hinge"},
         "apply_to_axes": ["colorize", "sharpen"]},
    ],
}
```

- [ ] **Step 4: Run test + full preset-loading suite to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loss_presets.py tests/test_config.py tests/test_configs_load.py -v`
Expected: all pass, including the new tests.

- [ ] **Step 5: Commit**

```bash
git add src/refine/config.py tests/test_loss_presets.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: rebalance loss presets — chroma_lab anchors hue, colorfulness weakened

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Per-factor SR preview rows (sharpen-2x/4x/8x)

**Why:** Current `sharpen-only` preview row samples factor∈{2,4,8} uniformly per render, so the user can't see whether the model handles different SR scales. Three explicit rows give immediate visual feedback per factor.

**Files:**
- Modify: `src/refine/train/trainer.py` (`_build_preview_samples`)
- Create: `tests/test_preview_sr_factors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_preview_sr_factors.py`:

```python
"""Preview must contain per-factor SR rows so the user sees sr2x/sr4x/sr8x."""
from __future__ import annotations

import torch

from restora_models.config import (
    RefineConfig, RunConfig, ModelConfig, DataConfig, LoaderConfig,
    AugmentConfig, CompoundConfig, LossConfig, OptimConfig,
    SchedulerConfig, TrainConfig, ExportConfig,
)


def _minimal_cfg(root: str) -> RefineConfig:
    return RefineConfig(
        run=RunConfig(name="test", output_dir=root, seed=0),
        model=ModelConfig(type="nafnet", size="tiny", input_size=64),
        data=DataConfig(
            root=root,
            val_fraction=0.5,
            num_fixed_preview_samples=1,
            num_random_preview_samples=0,
            augment=AugmentConfig(),
            loader=LoaderConfig(batch_size=2, num_workers=0,
                                pin_memory=False, persistent_workers=False),
        ),
        compound=CompoundConfig(),
        losses=[LossConfig(name="l1_rgb", weight=1.0)],
        optim_g=OptimConfig(), optim_d=OptimConfig(),
        scheduler=SchedulerConfig(total_steps=1),
        train=TrainConfig(total_steps=1),
        export=ExportConfig(on_finish=False),
    )


def test_preview_includes_per_factor_sr_rows(tmp_image_dir, monkeypatch):
    # We don't run training; we only instantiate a Trainer and call
    # _build_preview_samples directly.
    from restora_models.train.trainer import Trainer

    cfg = _minimal_cfg(str(tmp_image_dir))
    trainer = Trainer(cfg)
    samples = trainer._build_preview_samples()
    keys = list(samples.keys())
    assert "sharpen-2x" in keys
    assert "sharpen-4x" in keys
    assert "sharpen-8x" in keys
    # The legacy single "sharpen-only" row is gone (replaced by the three above).
    assert "sharpen-only" not in keys


def test_preview_per_factor_uses_different_factors(tmp_image_dir):
    """The degraded image in sharpen-2x and sharpen-8x must differ — different
    downsample factors yield different blur amounts."""
    from restora_models.train.trainer import Trainer

    cfg = _minimal_cfg(str(tmp_image_dir))
    trainer = Trainer(cfg)
    samples = trainer._build_preview_samples()
    deg2 = samples["sharpen-2x"][0]["degraded"]
    deg8 = samples["sharpen-8x"][0]["degraded"]
    # 8x is much blurrier → noticeable MSE vs the 2x version on the same source
    diff = (deg2 - deg8).abs().mean().item()
    assert diff > 1e-3, f"factors produced identical degraded images: diff={diff}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_preview_sr_factors.py -v`
Expected: FAIL — `"sharpen-2x"` not in samples keys (only `"sharpen-only"` exists).

- [ ] **Step 3: Update `_build_preview_samples`**

Edit `src/refine/train/trainer.py`. Find `_build_preview_samples` (~line 306). Replace its `preview_configs` block and the degradation logic with per-factor SR support:

```python
    @torch.inference_mode()
    def _build_preview_samples(self) -> dict[str, list[dict]]:
        n_fixed = self.cfg.data.num_fixed_preview_samples
        n_rand = self.cfg.data.num_random_preview_samples
        eval_model = self.ema.module if self.ema is not None else self.model
        was_training = eval_model.training
        eval_model.train(False)

        # 9 rows: identity + colorize + denoise + 3x sharpen + dejpeg + deblur + all-on.
        # The three sharpen rows force SR factors 2 / 4 / 8 so users see each scale.
        preview_configs: list[tuple[str, list[int], dict]] = [
            ("identity",      [0, 0, 0, 0, 0], {}),
            ("colorize-only", [1, 0, 0, 0, 0], {}),
            ("denoise-only",  [0, 1, 0, 0, 0], {}),
            ("sharpen-2x",    [0, 0, 1, 0, 0], {"sharpen_factor": 2}),
            ("sharpen-4x",    [0, 0, 1, 0, 0], {"sharpen_factor": 4}),
            ("sharpen-8x",    [0, 0, 1, 0, 0], {"sharpen_factor": 8}),
            ("dejpeg-only",   [0, 0, 0, 1, 0], {}),
            ("deblur-only",   [0, 0, 0, 0, 1], {}),
            ("all-on",        [1, 1, 1, 1, 1], {}),
        ]

        out: dict[str, list[dict]] = {label: [] for label, _, _ in preview_configs}
        n_total = len(self.val_ds.clean)
        idxs = list(range(min(n_fixed, n_total)))
        if n_rand > 0 and n_total > len(idxs):
            extra = torch.randint(len(idxs), n_total, (min(n_rand, n_total - len(idxs)),)).tolist()
            idxs += extra

        import random as _random
        from restora_models.data.degradations.registry import build_degradation
        _DEGRADE_ORDER = ("deblur", "denoise", "sharpen", "dejpeg", "colorize")
        _AXIS_TO_REG = {
            "colorize": "colorize", "denoise": "denoise",
            "sharpen": "sharpen", "dejpeg": "jpeg", "deblur": "deblur",
        }

        for label, vec, opts in preview_configs:
            flags = dict(zip(AXES, vec))
            # When we want a fixed SR factor, build a dedicated SharpenDegradation
            # with single-element factor_choices and use it in place of the
            # dataset's stochastic one for this row only.
            sharpen_override = None
            if "sharpen_factor" in opts:
                sharpen_override = build_degradation(
                    "sharpen", {"factor_choices": [int(opts["sharpen_factor"])]}
                )

            for i in idxs:
                clean_t = self.val_ds.clean[i]
                rng = _random.Random((self.cfg.run.seed * 1_000_003) ^ i)
                rgb_np = clean_t.permute(1, 2, 0).numpy().copy()
                for axis in _DEGRADE_ORDER:
                    if flags[axis]:
                        deg = sharpen_override if (axis == "sharpen" and sharpen_override is not None) \
                              else self.val_ds.degs[axis]
                        rgb_np = deg.degrade(rgb_np, rng)
                degraded_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).contiguous()
                cfg_t = torch.tensor([vec], dtype=torch.float32, device=self.device)
                pred = eval_model(degraded_t.unsqueeze(0).to(self.device), cfg_t)
                out[label].append({
                    "clean": clean_t, "degraded": degraded_t,
                    "predicted": pred.clamp(0, 1).squeeze(0).cpu(),
                })

        if was_training:
            eval_model.train(True)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_preview_sr_factors.py -v`
Expected: both pass.

Also run the existing preview test to confirm no regression:
Run: `.venv/bin/python -m pytest tests/test_preview.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/refine/train/trainer.py tests/test_preview_sr_factors.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: per-factor SR preview rows (sharpen-2x/4x/8x)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add PromptIR override fields to `ModelConfig`

**Why first in the PromptIR sequence:** All PromptIR code reads these fields. Decoupling the schema change from the model code lets Tasks 5–7 import a stable `ModelConfig` regardless of order.

**Files:**
- Modify: `src/refine/config.py` (find `class ModelConfig` ~line 14)
- Create: `tests/test_config_promptir_fields.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_promptir_fields.py`:

```python
from restora_models.config import ModelConfig


def test_promptir_fields_default_to_none():
    m = ModelConfig(type="promptir", size="large")
    assert m.prompt_n is None
    assert m.prompt_dim is None
    assert m.prompt_hw is None


def test_promptir_fields_accept_int_overrides():
    m = ModelConfig(type="promptir", size="tiny", prompt_n=7,
                    prompt_dim=48, prompt_hw=8)
    assert m.prompt_n == 7
    assert m.prompt_dim == 48
    assert m.prompt_hw == 8


def test_nafnet_unaffected():
    m = ModelConfig(type="nafnet", size="tiny", nf=32)
    assert m.nf == 32
    assert m.prompt_n is None
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_config_promptir_fields.py -v`
Expected: FAIL — `ModelConfig` rejects unknown fields `prompt_n`, etc.

- [ ] **Step 3: Add the fields**

Edit `src/refine/config.py`. Find the existing `class ModelConfig(BaseModel)` block and append three fields under the NAFNet ones (preserve all existing fields exactly):

```python
class ModelConfig(BaseModel):
    type: str = "nafnet"
    size: Literal["tiny", "large"] = "tiny"
    input_size: int = 256
    nf: int | None = None
    enc_depths: list[int] | None = None
    bottle_blocks: int | None = None
    hidden_dim: int | None = None
    task_embed_dim: int = 128
    # PromptIR-specific overrides (ignored when type != "promptir"):
    prompt_n: int | None = None
    prompt_dim: int | None = None
    prompt_hw: int | None = None
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_config_promptir_fields.py tests/test_config.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/refine/config.py tests/test_config_promptir_fields.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: ModelConfig — add prompt_n/dim/hw override fields for PromptIR

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: RestormerBlock (MDTA + GDFN + AdaLN)

**Why:** Building block for PromptIR. Replaces our existing TransformerBlock for the new backbone; existing one stays for NAFNet.

**Files:**
- Create: `src/refine/models/restormer_block.py`
- Create: `tests/test_restormer_block.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_restormer_block.py`:

```python
import torch

from restora_models.models.restormer_block import RestormerBlock


def test_forward_shape():
    blk = RestormerBlock(c=16, num_heads=2, task_dim=8)
    x = torch.randn(2, 16, 32, 32)
    task = torch.randn(2, 8)
    out = blk(x, task)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward():
    blk = RestormerBlock(c=8, num_heads=2, task_dim=4)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    task = torch.randn(1, 4)
    out = blk(x, task)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for n, p in blk.named_parameters():
        assert p.grad is not None, f"{n} got no gradient"


def test_adaln_changes_output():
    """AdaLN init is zero so output should equal x at init. Once we *bump*
    the AdaLN weights manually, different task vectors must produce different
    outputs (proving the conditioning is wired)."""
    blk = RestormerBlock(c=8, num_heads=2, task_dim=4)
    with torch.no_grad():
        for adaln in (blk.adaln1, blk.adaln2):
            adaln.weight.normal_(0, 0.1)
            adaln.bias.normal_(0, 0.1)
    x = torch.randn(1, 8, 16, 16)
    t1 = torch.zeros(1, 4)
    t2 = torch.ones(1, 4) * 5.0
    out1 = blk(x, t1); out2 = blk(x, t2)
    assert (out1 - out2).abs().mean().item() > 1e-3


def test_handles_non_square():
    blk = RestormerBlock(c=8, num_heads=2, task_dim=4)
    x = torch.randn(1, 8, 24, 40)
    out = blk(x, torch.zeros(1, 4))
    assert out.shape == x.shape
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_restormer_block.py -v`
Expected: FAIL — `RestormerBlock` not importable.

- [ ] **Step 3: Implement**

Create `src/refine/models/restormer_block.py`:

```python
"""Restormer transformer block: MDTA + GDFN, with AdaLN modulation.

References:
- Restormer (Zamir et al., CVPR 2022) — MDTA + GDFN.
- DiT (Peebles & Xie, 2023) — AdaLN-style scalar modulation from a
  conditioning vector. We reuse this for our 5-axis config conditioning
  so a single ConfigEmbed feeds every block.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNormChan(nn.Module):
    """LayerNorm over the channel axis for (B, C, H, W) tensors."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-5)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class MDTA(nn.Module):
    """Multi-Dconv head Transposed Attention. Attention is computed along
    the channel axis (each head sees C/h channels), so cost is O((C/h)^2 * HW)
    instead of vanilla self-attention's O((HW)^2)."""

    def __init__(self, c: int, num_heads: int) -> None:
        super().__init__()
        assert c % num_heads == 0, f"channels {c} not divisible by heads {num_heads}"
        self.num_heads = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, kernel_size=3, padding=1,
                                groups=c * 3, bias=False)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temp
        # FP32 softmax for numerical stability under bf16/fp16 training
        attn = attn.float().softmax(dim=-1).to(v.dtype)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward."""

    def __init__(self, c: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(round(c * expansion))
        self.proj_in = nn.Conv2d(c, hidden * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                            groups=hidden * 2, bias=False)
        self.proj_out = nn.Conv2d(hidden, c, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.dw(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(a) * b)


class RestormerBlock(nn.Module):
    def __init__(self, *, c: int, num_heads: int, task_dim: int,
                 ffn_expansion: float = 2.66) -> None:
        super().__init__()
        self.norm1 = LayerNormChan(c)
        self.attn  = MDTA(c, num_heads=num_heads)
        self.norm2 = LayerNormChan(c)
        self.ffn   = GDFN(c, expansion=ffn_expansion)
        self.adaln1 = nn.Linear(task_dim, 2 * c)
        self.adaln2 = nn.Linear(task_dim, 2 * c)
        # Zero-init AdaLN projections so the block starts at identity
        # modulation (scale=1, shift=0). Standard DiT trick.
        nn.init.zeros_(self.adaln1.weight); nn.init.zeros_(self.adaln1.bias)
        nn.init.zeros_(self.adaln2.weight); nn.init.zeros_(self.adaln2.bias)

    def _mod(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(b, c, 1, 1); beta = beta.view(b, c, 1, 1)
        return x * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self._mod(self.norm1(x), self.adaln1(task_vec)))
        x = x + self.ffn (self._mod(self.norm2(x), self.adaln2(task_vec)))
        return x
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_restormer_block.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/refine/models/restormer_block.py tests/test_restormer_block.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: add RestormerBlock (MDTA + GDFN + AdaLN) for PromptIR backbone

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: PromptBlock (config-driven prompt mix)

**Files:**
- Create: `src/refine/models/prompt_block.py`
- Create: `tests/test_prompt_block.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt_block.py`:

```python
import torch

from restora_models.models.prompt_block import PromptBlock


def test_forward_shape():
    blk = PromptBlock(feat_c=16, prompt_n=5, prompt_dim=16, prompt_hw=8, cond_dim=8)
    x = torch.randn(2, 16, 32, 32)
    cond = torch.randn(2, 8)
    out = blk(x, cond)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_config_determinism_same_config_same_output():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    blk.train(False)
    x = torch.randn(1, 8, 16, 16)
    c = torch.tensor([[1.0, 0, 0, 0]])
    o1 = blk(x, c); o2 = blk(x, c)
    assert torch.allclose(o1, o2)


def test_different_configs_different_outputs():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    with torch.no_grad():
        for i in range(blk.prompts.shape[0]):
            blk.prompts[i].fill_(float(i + 1) * 0.5)
    x = torch.randn(1, 8, 16, 16)
    c1 = torch.tensor([[5.0, -5, -5, -5]])
    c2 = torch.tensor([[-5, 5.0, -5, -5]])
    o1 = blk(x, c1); o2 = blk(x, c2)
    assert (o1 - o2).abs().mean().item() > 1e-4


def test_backward_grads_all_params():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    cond = torch.randn(1, 4)
    out = blk(x, cond)
    out.pow(2).mean().backward()
    for n, p in blk.named_parameters():
        assert p.grad is not None, f"{n} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{n} has non-finite grad"


def test_handles_non_square_feat():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    x = torch.randn(1, 8, 24, 40)
    out = blk(x, torch.zeros(1, 4))
    assert out.shape == x.shape
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_prompt_block.py -v`
Expected: FAIL — `PromptBlock` not importable.

- [ ] **Step 3: Implement**

Create `src/refine/models/prompt_block.py`:

```python
"""Config-driven PromptBlock.

Replaces PromptIR's paper-original blind self-attention prompt-selection
with a router driven by the 5-axis config embedding. The same `cond`
that drives every AdaLN in the network also picks which learned prompts
to mix here.

Property: identical config → identical mix → identical output.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PromptBlock(nn.Module):
    def __init__(self, *, feat_c: int, prompt_n: int = 5,
                 prompt_dim: int, prompt_hw: int, cond_dim: int) -> None:
        super().__init__()
        # Learnable prompt bank, shape (N, prompt_dim, P_h, P_w).
        # Small Gaussian init so different prompts start at different signals.
        self.prompts = nn.Parameter(
            torch.randn(prompt_n, prompt_dim, prompt_hw, prompt_hw) * 0.02
        )
        self.router = nn.Linear(cond_dim, prompt_n)
        self.fuse = nn.Conv2d(feat_c + prompt_dim, feat_c, kernel_size=1)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        alpha = F.softmax(self.router(cond), dim=-1)            # (B, N)
        mix = (alpha[:, :, None, None, None]
               * self.prompts.unsqueeze(0)).sum(dim=1)          # (B, P_c, P_h, P_w)
        mix = F.interpolate(mix, size=(h, w), mode="bilinear",
                            align_corners=False)
        return self.fuse(torch.cat([feat, mix], dim=1))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_prompt_block.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/refine/models/prompt_block.py tests/test_prompt_block.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: add config-driven PromptBlock for PromptIR backbone

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: PromptIR backbone (compose Restormer + PromptBlock)

**Files:**
- Create: `src/refine/models/promptir.py`
- Modify: `src/refine/models/__init__.py` (add import)
- Create: `tests/test_promptir.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_promptir.py`:

```python
import os

import pytest
import torch

from restora_models.config import ModelConfig
from restora_models.models import build_model
from restora_models.models.registry import MODEL_REGISTRY


def test_promptir_registered():
    assert "promptir" in MODEL_REGISTRY


@pytest.mark.parametrize("size", ["tiny", "large"])
def test_forward_shape(size):
    cfg = ModelConfig(type="promptir", size=size, input_size=64)
    m = build_model(cfg, num_axes=5)
    x = torch.randn(2, 3, 64, 64)
    c = torch.zeros(2, 5)
    out = m(x, c)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward_smoke_reduces_loss():
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    target = torch.rand(1, 3, 32, 32)
    x = torch.randn(1, 3, 32, 32)
    c = torch.tensor([[1.0, 0, 0, 0, 0]])
    losses = []
    for _ in range(3):
        out = m(x, c)
        loss = (out - target).abs().mean()
        losses.append(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_param_count_sane():
    tiny  = build_model(ModelConfig(type="promptir", size="tiny",  input_size=64), num_axes=5)
    large = build_model(ModelConfig(type="promptir", size="large", input_size=64), num_axes=5)
    n_tiny  = sum(p.numel() for p in tiny.parameters())
    n_large = sum(p.numel() for p in large.parameters())
    assert 1_000_000 < n_tiny < 15_000_000, f"tiny param count {n_tiny}"
    assert 10_000_000 < n_large < 60_000_000, f"large param count {n_large}"


def test_identity_config_passes_input_through():
    """With config=0 and the model's zero-init head, residual carries identity."""
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    x = torch.rand(1, 3, 32, 32)
    c = torch.zeros(1, 5)
    with torch.no_grad():
        out = m(x, c)
    diff = (out - x).abs().mean().item()
    assert diff < 0.1, f"identity output deviated from input: diff={diff}"


def test_different_configs_different_outputs():
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    x = torch.rand(1, 3, 32, 32)
    c1 = torch.tensor([[1.0, 0, 0, 0, 0]])
    c2 = torch.tensor([[0, 0, 0, 0, 1.0]])
    with torch.no_grad():
        o1 = m(x, c1); o2 = m(x, c2)
    # Random prompt init means even untrained, prompt mixes diverge.
    assert (o1 - o2).abs().mean().item() > 1e-5


@pytest.mark.skipif(not os.environ.get("REFINE_SLOW"), reason="slow ONNX export, set REFINE_SLOW=1")
def test_onnx_export_parity_all_configs(tmp_path):
    from restora_models.export.onnx import export_onnx_from_model

    cfg = ModelConfig(type="promptir", size="tiny", input_size=64)
    m = build_model(cfg, num_axes=5)
    out = tmp_path / "promptir.onnx"
    export_onnx_from_model(m, num_axes=5, input_size=64, export_path=out,
                           opset=17, simplify=False, verify_parity=True,
                           parity_atol=1e-3, dynamic_hw=False,
                           task_map={"model_type": "promptir"})
    assert out.exists()
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_promptir.py -v`
Expected: FAIL — `promptir` not in `MODEL_REGISTRY` (or import error).

- [ ] **Step 3: Implement the backbone**

Create `src/refine/models/promptir.py`:

```python
"""PromptIR backbone — config-driven prompt variant.

4-level Restormer-style U-Net with config-driven PromptBlocks
interleaved on the decoder path. Same forward contract as NAFNet:
forward(rgb: (B,3,H,W) in [0,1], config: (B,5) float) -> (B,3,H,W).
"""
from __future__ import annotations

import torch
from torch import nn

from restora_models.config import ModelConfig
from .prompt_block import PromptBlock
from .registry import register_model
from .restormer_block import RestormerBlock
from .task_embed import ConfigEmbed


_SIZE_PRESETS: dict[str, dict] = {
    "tiny": {
        "dim": 24, "depths": [2, 2, 2, 2], "refinement": 2,
        "heads": [1, 2, 4, 8],
        "prompt_n": 5, "prompt_dim": 32, "prompt_hw": 8,
    },
    "large": {
        "dim": 48, "depths": [4, 6, 6, 8], "refinement": 4,
        "heads": [1, 2, 4, 8],
        "prompt_n": 5, "prompt_dim": 64, "prompt_hw": 16,
    },
}


def _resolve(cfg: ModelConfig) -> dict:
    preset = _SIZE_PRESETS[cfg.size]
    return {
        "dim":         preset["dim"],
        "depths":      preset["depths"],
        "refinement":  preset["refinement"],
        "heads":       preset["heads"],
        "task_dim":    cfg.task_embed_dim,
        "prompt_n":    cfg.prompt_n   if cfg.prompt_n   is not None else preset["prompt_n"],
        "prompt_dim":  cfg.prompt_dim if cfg.prompt_dim is not None else preset["prompt_dim"],
        "prompt_hw":   cfg.prompt_hw  if cfg.prompt_hw  is not None else preset["prompt_hw"],
    }


def _stack(c: int, n: int, num_heads: int, task_dim: int) -> nn.ModuleList:
    return nn.ModuleList(
        [RestormerBlock(c=c, num_heads=num_heads, task_dim=task_dim) for _ in range(n)]
    )


@register_model("promptir")
class PromptIR(nn.Module):
    def __init__(self, cfg: ModelConfig, *, num_axes: int = 5) -> None:
        super().__init__()
        p = _resolve(cfg)
        dim  = p["dim"]; depths = p["depths"]; heads = p["heads"]
        ref_n = p["refinement"]; task_dim = p["task_dim"]
        assert len(depths) == 4
        assert len(heads) == 4

        self.task_embed = ConfigEmbed(num_axes=num_axes, dim=task_dim)

        # Stem
        self.stem = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        # Encoder
        self.enc_l1 = _stack(dim,       depths[0], heads[0], task_dim)
        self.down1  = nn.PixelUnshuffle(2); self.down1_proj = nn.Conv2d(dim * 4,     dim * 2, 1)
        self.enc_l2 = _stack(dim * 2,   depths[1], heads[1], task_dim)
        self.down2  = nn.PixelUnshuffle(2); self.down2_proj = nn.Conv2d(dim * 2 * 4, dim * 4, 1)
        self.enc_l3 = _stack(dim * 4,   depths[2], heads[2], task_dim)
        self.down3  = nn.PixelUnshuffle(2); self.down3_proj = nn.Conv2d(dim * 4 * 4, dim * 8, 1)
        self.latent = _stack(dim * 8,   depths[3], heads[3], task_dim)

        # Decoder: PromptBlock → upsample → skip concat 1x1 → transformer stack
        self.prompt_l3 = PromptBlock(
            feat_c=dim * 8, prompt_n=p["prompt_n"],
            prompt_dim=p["prompt_dim"], prompt_hw=p["prompt_hw"], cond_dim=task_dim,
        )
        self.up3       = nn.PixelShuffle(2)
        self.up3_proj  = nn.Conv2d(dim * 2,     dim * 4, 1)
        self.skip3     = nn.Conv2d(dim * 4 * 2, dim * 4, 1)
        self.dec_l3    = _stack(dim * 4,   depths[2], heads[2], task_dim)

        self.prompt_l2 = PromptBlock(
            feat_c=dim * 4, prompt_n=p["prompt_n"],
            prompt_dim=p["prompt_dim"], prompt_hw=p["prompt_hw"], cond_dim=task_dim,
        )
        self.up2       = nn.PixelShuffle(2)
        self.up2_proj  = nn.Conv2d(dim,         dim * 2, 1)
        self.skip2     = nn.Conv2d(dim * 2 * 2, dim * 2, 1)
        self.dec_l2    = _stack(dim * 2,   depths[1], heads[1], task_dim)

        self.prompt_l1 = PromptBlock(
            feat_c=dim * 2, prompt_n=p["prompt_n"],
            prompt_dim=p["prompt_dim"], prompt_hw=p["prompt_hw"], cond_dim=task_dim,
        )
        self.up1       = nn.PixelShuffle(2)
        self.up1_proj  = nn.Conv2d(dim // 2,    dim, 1)
        self.skip1     = nn.Conv2d(dim * 2,     dim, 1)
        self.dec_l1    = _stack(dim,       depths[0], heads[0], task_dim)

        self.refinement = _stack(dim, ref_n, heads[0], task_dim)

        # Output head — zero-init so initial output = input via global residual
        self.head = nn.Conv2d(dim, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _run(stack: nn.ModuleList, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        for blk in stack:
            x = blk(x, task)
        return x

    def forward(self, rgb: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        task = self.task_embed(config)

        x = self.stem(rgb)

        e1 = self._run(self.enc_l1, x, task)
        e2 = self._run(self.enc_l2, self.down1_proj(self.down1(e1)), task)
        e3 = self._run(self.enc_l3, self.down2_proj(self.down2(e2)), task)
        lat = self._run(self.latent, self.down3_proj(self.down3(e3)), task)

        d = self.prompt_l3(lat, task)
        d = self.up3_proj(self.up3(d))
        d = self.skip3(torch.cat([d, e3], dim=1))
        d = self._run(self.dec_l3, d, task)

        d = self.prompt_l2(d, task)
        d = self.up2_proj(self.up2(d))
        d = self.skip2(torch.cat([d, e2], dim=1))
        d = self._run(self.dec_l2, d, task)

        d = self.prompt_l1(d, task)
        d = self.up1_proj(self.up1(d))
        d = self.skip1(torch.cat([d, e1], dim=1))
        d = self._run(self.dec_l1, d, task)

        d = self._run(self.refinement, d, task)
        return rgb + self.head(d)
```

Modify `src/refine/models/__init__.py` — add this line near the existing model registration imports (the file already imports `nafnet`; add a parallel import):

```python
from . import promptir as _promptir  # noqa: F401  registers "promptir"
```

(If the existing `__init__.py` doesn't yet import `nafnet`, read the file to confirm how registration happens, then add the import in the same pattern.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_promptir.py -v`
Expected: 6 passed, 1 skipped (the ONNX parity test is gated by `REFINE_SLOW`).

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_promptir.py::test_onnx_export_parity_all_configs -v`
Expected: PASS within ~30 s.

- [ ] **Step 5: Commit**

```bash
git add src/refine/models/promptir.py src/refine/models/__init__.py tests/test_promptir.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: PromptIR backbone (config-driven prompt variant)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: fp16 ONNX export

**Why:** Highest-value precision option, broadly supported across runtimes/GPUs. Implementing the precision parameter on the existing exporter also paves the way for fp8/fp4 in the next tasks (fp8 lives in the same exporter, fp4 is a stub).

**Files:**
- Modify: `src/refine/export/onnx.py`
- Modify: `src/refine/cli.py` (add `--precision` flag)
- Create: `tests/test_export_precision.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_precision.py`:

```python
import os

import pytest
import torch

from restora_models.config import ModelConfig
from restora_models.export.onnx import export_onnx_from_model
from restora_models.models import build_model


@pytest.mark.skipif(not os.environ.get("REFINE_SLOW"), reason="slow ONNX export, set REFINE_SLOW=1")
def test_fp16_export_round_trip(tmp_path):
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model_fp16.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=32, export_path=out,
        opset=17, simplify=False, verify_parity=True, parity_atol=5e-2,
        dynamic_hw=False, task_map={"model_type": "nafnet"},
        precision="fp16",
    )
    assert out.exists()
    import onnx
    om = onnx.load(str(out))
    fp16 = sum(1 for init in om.graph.initializer if init.data_type == onnx.TensorProto.FLOAT16)
    fp32 = sum(1 for init in om.graph.initializer if init.data_type == onnx.TensorProto.FLOAT)
    assert fp16 > 0
    assert fp16 > fp32, f"expected fp16 dominance; got fp16={fp16} fp32={fp32}"


def test_fp8_raises_capability_error_on_unsupported_runtime(tmp_path):
    """If the local onnxruntime build lacks fp8 support, fp8 export must raise a
    clear error message naming the missing capability — not silently fall back."""
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model_fp8.onnx"
    try:
        export_onnx_from_model(
            m, num_axes=5, input_size=32, export_path=out,
            opset=19, simplify=False, verify_parity=False,
            dynamic_hw=False, task_map=None, precision="fp8",
        )
    except (NotImplementedError, RuntimeError) as e:
        msg = str(e).lower()
        assert ("fp8" in msg) or ("e4m3" in msg) or ("opset" in msg) or ("not supported" in msg)


def test_fp4_raises_not_implemented(tmp_path):
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model_fp4.onnx"
    with pytest.raises(NotImplementedError) as ei:
        export_onnx_from_model(
            m, num_axes=5, input_size=32, export_path=out,
            opset=21, simplify=False, verify_parity=False,
            dynamic_hw=False, task_map=None, precision="fp4",
        )
    assert "fp4" in str(ei.value).lower() or "nvfp4" in str(ei.value).lower()
    assert "tensorrt" in str(ei.value).lower() or "modelopt" in str(ei.value).lower()


def test_invalid_precision_rejected(tmp_path):
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model.onnx"
    with pytest.raises(ValueError):
        export_onnx_from_model(
            m, num_axes=5, input_size=32, export_path=out,
            opset=17, simplify=False, verify_parity=False,
            precision="fp64",
        )
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_export_precision.py -v`
Expected: FAIL — `precision=` parameter not supported on `export_onnx_from_model`.

- [ ] **Step 3: Implement precision parameter**

Replace the contents of `src/refine/export/onnx.py` with:

```python
"""Export refine compound model to ONNX with per-config parity verification.

Supports precision={"fp32" (default), "fp16", "fp8", "fp4"}. fp16 is a post-export
conversion via onnxconverter-common. fp8 attempts post-training quantization via
onnxruntime.quantization (requires opset 19+ and ORT 1.17+); on unsupported
runtimes it raises with a clear message. fp4 raises NotImplementedError pointing
at TensorRT 10+ / NVIDIA modelopt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn


Precision = Literal["fp32", "fp16", "fp8", "fp4"]
_VALID_PRECISION = ("fp32", "fp16", "fp8", "fp4")


def _reference_configs(num_axes: int) -> list[tuple[str, list[float]]]:
    configs = [("identity", [0.0] * num_axes), ("all-on", [1.0] * num_axes)]
    for i in range(num_axes):
        v = [0.0] * num_axes
        v[i] = 1.0
        configs.append((f"axis{i}-only", v))
    return configs


def _convert_to_fp16(path: Path) -> None:
    try:
        from onnxconverter_common import float16
    except ImportError as e:
        raise RuntimeError(
            "fp16 export requires `onnxconverter-common`; install with "
            "`uv pip install onnxconverter-common`"
        ) from e
    import onnx
    m = onnx.load(str(path))
    m_fp16 = float16.convert_float_to_float16(
        m, keep_io_types=False, disable_shape_infer=False,
    )
    onnx.save(m_fp16, str(path))


def _quantize_to_fp8(path: Path, opset: int) -> None:
    if opset < 19:
        raise RuntimeError(f"fp8 requires ONNX opset >= 19; got opset={opset}")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise RuntimeError(
            "fp8 export requires onnxruntime>=1.17 with quantization support"
        ) from e
    if not hasattr(QuantType, "QFloat8E4M3FN"):
        raise RuntimeError(
            "Local onnxruntime build lacks fp8 (E4M3) QuantType — "
            "upgrade to onnxruntime>=1.17 with cuda12-fp8 support"
        )
    tmp = path.with_suffix(path.suffix + ".pre-fp8")
    os.replace(path, tmp)
    quantize_dynamic(
        model_input=str(tmp), model_output=str(path),
        weight_type=QuantType.QFloat8E4M3FN,
    )
    os.remove(tmp)


def export_onnx_from_model(
    model: nn.Module, *,
    num_axes: int,
    input_size: int,
    export_path: str | Path,
    opset: int = 17,
    simplify: bool = True,
    verify_parity: bool = True,
    parity_atol: float = 1e-3,
    dynamic_hw: bool = False,
    task_map: dict | None = None,
    precision: Precision = "fp32",
) -> None:
    if precision not in _VALID_PRECISION:
        raise ValueError(f"unknown precision {precision!r}; must be one of {_VALID_PRECISION}")
    if precision == "fp4":
        raise NotImplementedError(
            "fp4 / NVFP4 export not yet supported by stable tooling. "
            "Requires TensorRT 10+ on a Blackwell-class GPU (B100/B200/GB200); "
            "see NVIDIA modelopt (https://github.com/NVIDIA/TensorRT-Model-Optimizer) "
            "for the current path. This stub will be replaced once onnxruntime "
            "gains stable fp4 support."
        )

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.train(False)

    dummy_rgb = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    dummy_cfg = torch.zeros(1, num_axes, dtype=torch.float32)

    dynamic_axes: dict[str, dict[int, str]] = {
        "input":  {0: "batch"},
        "config": {0: "batch"},
        "output": {0: "batch"},
    }
    if dynamic_hw:
        dynamic_axes["input"][2] = "height"; dynamic_axes["input"][3] = "width"
        dynamic_axes["output"][2] = "height"; dynamic_axes["output"][3] = "width"

    torch.onnx.export(
        model, (dummy_rgb, dummy_cfg), str(export_path),
        opset_version=opset,
        input_names=["input", "config"], output_names=["output"],
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

    if precision == "fp16":
        _convert_to_fp16(export_path)
    elif precision == "fp8":
        _quantize_to_fp8(export_path, opset=opset)

    if verify_parity:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        for label, vec in _reference_configs(num_axes):
            x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
            c = np.array([vec], dtype=np.float32)
            ort_out = sess.run(None, {"input": x, "config": c})[0]
            with torch.no_grad():
                t_out = model(torch.from_numpy(x), torch.from_numpy(c)).numpy()
            diff = float(np.abs(ort_out - t_out).max())
            if diff > parity_atol:
                raise RuntimeError(
                    f"ONNX parity failed for {label} ({precision}): max_abs_diff={diff:.3e}"
                )
        if dynamic_hw:
            alt_h = max(48, input_size // 2); alt_w = max(48, input_size // 2 + 32)
            for label, vec in _reference_configs(num_axes):
                x = np.random.rand(1, 3, alt_h, alt_w).astype(np.float32)
                c = np.array([vec], dtype=np.float32)
                try:
                    ort_out = sess.run(None, {"input": x, "config": c})[0]
                except Exception as e:
                    raise RuntimeError(f"dynamic_hw ONNX rejected {alt_h}x{alt_w}: {e}") from e
                with torch.no_grad():
                    t_out = model(torch.from_numpy(x), torch.from_numpy(c)).numpy()
                diff = float(np.abs(ort_out - t_out).max())
                if diff > parity_atol:
                    raise RuntimeError(
                        f"dynamic-hw parity failed for {label} at {alt_h}x{alt_w} ({precision}): "
                        f"max_abs_diff={diff:.3e}")

    if task_map is not None:
        sidecar = export_path.with_suffix(".task_map.json")
        task_map_with_prec = dict(task_map)
        task_map_with_prec["precision"] = precision
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(task_map_with_prec, indent=2))
        os.replace(sidecar_tmp, sidecar)
```

Modify `src/refine/cli.py`'s `export` command. Replace its signature + body with:

```python
@app.command()
def export(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(256, "--input-size"),
    opset: int = typer.Option(17, "--opset"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify"),
    dynamic_hw: bool = typer.Option(False, "--dynamic-hw/--fixed-hw"),
    precision: str = typer.Option("fp32", "--precision",
                                  help="fp32 (default) | fp16 | fp8 | fp4"),
) -> None:
    import torch
    from restora_models.config import ModelConfig
    from restora_models.data.compound import AXES
    from restora_models.export.onnx import export_onnx_from_model
    from restora_models.models import build_model

    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg = ModelConfig(**(cfg_dict.get("model") or {}))
    m = build_model(mcfg, num_axes=len(AXES))
    m.load_state_dict(payload["model"])
    task_map = payload.get("task_map") or {}
    # fp8 implicitly requires opset 19 — bump if user left it at default
    effective_opset = max(opset, 19) if precision == "fp8" else opset
    export_onnx_from_model(
        m, num_axes=len(AXES), input_size=input_size,
        export_path=output, opset=effective_opset, simplify=simplify,
        dynamic_hw=dynamic_hw, task_map=task_map, precision=precision,
    )
    typer.echo(f"wrote {output} ({precision})")
```

Install `onnxconverter-common`:

```bash
.venv/bin/uv pip install onnxconverter-common
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_export_precision.py -v`
Expected: 3 passed (fp16 round-trip skipped without REFINE_SLOW, fp4 raises, invalid raises).

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_export_precision.py -v`
Expected: 4 passed.

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_export_onnx.py -v`
Expected: 1 passed (no regression on the original exporter).

- [ ] **Step 5: Commit**

```bash
git add src/refine/export/onnx.py src/refine/cli.py tests/test_export_precision.py pyproject.toml uv.lock
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: ONNX export --precision fp32/fp16/fp8/fp4 (fp4 stubbed with TRT pointer)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: PromptIR-laion production training config

**Files:**
- Create: `configs/promptir-laion.yaml`
- Modify or create: `tests/test_configs_load.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_configs_load.py` (if absent, create with this content):

```python
def test_promptir_laion_config_loads():
    from pathlib import Path
    from restora_models.config import load_config
    cfg = load_config(Path("configs/promptir-laion.yaml"))
    assert cfg.model.type == "promptir"
    assert cfg.model.size == "large"
    names = [l.name for l in cfg.losses]
    assert "chroma_lab" in names
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_configs_load.py::test_promptir_laion_config_loads -v`
Expected: FAIL — config file doesn't exist.

- [ ] **Step 3: Write the config**

Create `configs/promptir-laion.yaml`:

```yaml
defaults: default.yaml

run:
  name: "promptir-large-laion-${date:%Y-%m-%d_%H-%M-%S}"

model:
  type: promptir
  size: large
  input_size: 256
  task_embed_dim: 128

data:
  root: "~/data/laion-images"
  val_fraction: 0.005
  num_fixed_preview_samples: 1
  num_random_preview_samples: 1
  loader:
    batch_size: 8
    num_workers: 16
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4

compound:
  identity_prob: 0.05
  axis_probs:
    colorize: 0.5
    denoise:  0.5
    sharpen:  0.5
    dejpeg:   0.5
    deblur:   0.5
  degradations:
    colorize: {}
    denoise:  { sigma_range: [0.005, 0.05] }
    sharpen:  { factor_choices: [2, 4, 8] }
    dejpeg:   { quality_range: [20, 70] }
    deblur:   { sigma_range: [1.0, 3.0], motion_prob: 0.2 }

# Use the rebalanced "standard" preset that includes chroma_lab.
losses: !preset standard

optim_g:
  type: AdamW
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.99]
  fused: true

scheduler:
  type: cosine
  warmup_steps: 3000
  total_steps: 200000

train:
  total_steps: 200000
  amp: "bf16"
  preview_every_s: 60
  preview_history_every: 1000

export:
  on_finish: true
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_configs_load.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add configs/promptir-laion.yaml tests/test_configs_load.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: configs/promptir-laion.yaml — PromptIR-large production config

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: End-to-end smoke test (PromptIR train → ckpt → preview → infer → ONNX fp16)

**Why:** Single test that exercises the full pipeline against the new backbone + chroma loss + per-factor preview + fp16 export. If this passes, every change in the plan is integration-clean.

**Files:**
- Create: `tests/test_promptir_e2e_smoke.py`

- [ ] **Step 1: Write the test (sets the contract)**

Create `tests/test_promptir_e2e_smoke.py`:

```python
"""End-to-end smoke for PromptIR: train 10 steps → ckpt → preview → infer → ONNX fp16."""
from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not os.environ.get("REFINE_SLOW"),
    reason="full smoke test (~60s on CPU); set REFINE_SLOW=1 to run",
)


def test_promptir_full_pipeline(tmp_path, tmp_image_dir):
    from restora_models.config import (
        RefineConfig, RunConfig, ModelConfig, DataConfig, LoaderConfig,
        AugmentConfig, CompoundConfig, OptimConfig, SchedulerConfig,
        TrainConfig, ExportConfig, expand_loss_preset,
    )
    from restora_models.train.trainer import Trainer

    cfg = RefineConfig(
        run=RunConfig(name="smoke", output_dir=str(tmp_path), seed=0),
        model=ModelConfig(type="promptir", size="tiny", input_size=64),
        data=DataConfig(
            root=str(tmp_image_dir),
            val_fraction=0.25,
            num_fixed_preview_samples=1,
            num_random_preview_samples=0,
            augment=AugmentConfig(),
            loader=LoaderConfig(batch_size=2, num_workers=0,
                                pin_memory=False, persistent_workers=False),
        ),
        compound=CompoundConfig(),
        losses=expand_loss_preset("standard"),
        optim_g=OptimConfig(), optim_d=OptimConfig(),
        scheduler=SchedulerConfig(total_steps=10),
        train=TrainConfig(total_steps=10, amp="fp32",
                          preview_every_s=999999, preview_history_every=0),
        export=ExportConfig(on_finish=True),
    )
    trainer = Trainer(cfg)
    trainer.fit()

    # Final ckpt exists
    final = tmp_path / "ckpt" / "final.pt"
    assert final.exists(), f"no checkpoint at {final}"

    # Preview was written
    latest = tmp_path / "samples" / "latest.png"
    assert latest.exists(), "preview not written"

    # Sidecar JSON says promptir
    import json
    sidecar = final.with_suffix(".task_map.json")
    if sidecar.exists():
        tm = json.loads(sidecar.read_text())
        assert tm.get("model_type") == "promptir"

    # fp16 ONNX export round-trip
    from restora_models.export.onnx import export_onnx_from_model
    from restora_models.models import build_model

    payload = torch.load(str(final), map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**(payload["extra"]["cfg"]["model"]))
    m = build_model(mcfg, num_axes=5)
    m.load_state_dict(payload["model"])
    onnx_path = tmp_path / "model.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=64, export_path=onnx_path,
        opset=17, simplify=False, verify_parity=True, parity_atol=5e-2,
        dynamic_hw=False, task_map={"model_type": "promptir"},
        precision="fp16",
    )
    assert onnx_path.exists()

    # Sanity-infer one config through the saved fp16 ONNX
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 64, 64).astype(np.float16)
    c = np.array([[1.0, 0, 0, 0, 0]], dtype=np.float16)
    y = sess.run(None, {"input": x, "config": c})[0]
    assert y.shape == (1, 3, 64, 64)
    assert np.isfinite(y).all()
```

- [ ] **Step 2: Run to verify fail (or surface bugs)**

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_promptir_e2e_smoke.py -v`
Expected: FAIL on first action (build_model, Trainer init, fit, or export). Whichever assertion fires identifies the bug; fix in the relevant upstream task's files.

- [ ] **Step 3: No new implementation** — this test is a guardrail over Tasks 1–9.

- [ ] **Step 4: Verify pass**

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/test_promptir_e2e_smoke.py -v`
Expected: PASS in <2 minutes on CPU.

- [ ] **Step 5: Commit**

```bash
git add tests/test_promptir_e2e_smoke.py
git -c user.name=bglueck -c user.email=gluber1980@gmail.com commit -m "$(cat <<'EOF'
refine: e2e smoke test — PromptIR train→ckpt→preview→infer→fp16 ONNX

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final verification gate

**Files:** none changed; this is the gate before declaring the branch ready.

- [ ] **Step 1: Run full fast suite**

Run: `.venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass (slow tests skipped).

- [ ] **Step 2: Run full slow suite**

Run: `REFINE_SLOW=1 .venv/bin/python -m pytest tests/ -q --tb=short`
Expected: all pass.

- [ ] **Step 3: Sanity-check the CLI surface**

```bash
.venv/bin/refine --help
.venv/bin/refine export --help
```

Expected: `--precision` flag visible with help text.

- [ ] **Step 4: Confirm clean `git status`**

```bash
git status
git log --oneline -15
```

Expected: clean tree, 10 new commits ahead of `a2f6ea7`.

---

## Self-Review Notes

Spec coverage check (against `docs/superpowers/specs/2026-05-13-promptir-backbone-design.md`):

- §3 Architecture (4-level U-Net, encoder/decoder/latent/refinement, prompt block injection): Task 7.
- §4 PromptBlock with config-driven router: Task 6.
- §5 Conditioning via ConfigEmbed for both AdaLN and prompt router: Task 5 (AdaLN) + Task 6 (router).
- §6 Size presets tiny/large: Task 7 (`_SIZE_PRESETS`).
- §7 ModelConfig additions (`prompt_n/dim/hw`): Task 4.
- §8 File list: every file in §8 has a task that creates it.
- §9 Tests including ONNX parity: Tasks 5/6/7 plus the slow integration in Task 10.
- §10 AMP fp32 softmax in MDTA: Task 5 (`attn.float().softmax(dim=-1).to(v.dtype)`).
- §11 Inference / export / sidecar unchanged: Task 8 preserves the contract, just adds precision parameter.

User's additional asks coverage:
- Chroma loss: Tasks 1+2.
- SR preview per factor: Task 3.
- fp16/fp8/fp4 export: Task 8.
- promptir-laion config: Task 9.
- End-to-end smoke: Task 10.
- Verification gate: Task 11.

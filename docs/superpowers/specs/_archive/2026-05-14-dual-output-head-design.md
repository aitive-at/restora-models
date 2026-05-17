# Dual-Output Head + ONNX Export Wrapper

**Status:** Approved
**Date:** 2026-05-14
**Owner:** bglueck
**Lineage:** Addresses the colorize/sharpen quality gap surfaced after
the 2026-05-13 `nafnet-tiny-vivid` experiment (loss-weight rebalance
alone made colorize muddier, sharpen blurrier, and easy tasks regress
— a zero-sum capacity reallocation). Replaces the single 3-channel
output head with a dual head that recovers v1's structural advantage
for colorization while keeping the framework's `(rgb, config) → rgb`
contract bit-identical for downstream consumers.

## 1. Goal

Make colorize work as well in v3 as it did in v1 (legacy coliraz-v1)
without sacrificing the other axes. v1's secret was architectural, not
just a loss recipe: the DDColor model output the Lab `ab` channels
directly and merged with the input's `L` at the end. We restore that
property *as a second output head* — selectable per-sample by the
existing `colorize` axis flag — while keeping the existing RGB-delta
head for every other axis.

## 2. Non-goals

- Replacing the existing RGB-delta path for non-colorize axes.
- Changing the external `forward(rgb, config) → rgb` contract.
- Adding any new ONNX input or output tensor names.
- Changing the inference CLI / pipeline / sidecar JSON format.
- Reintroducing GAN losses (deferred until the structural fix is in).
- Curriculum / phased training (deferred — likely unnecessary if the
  architectural fix works).

## 3. Architecture — `DualOutputHead`

A new module lives in `src/refine/models/heads.py`:

```python
class DualOutputHead(nn.Module):
    """Two parallel output heads composed into a single RGB output.

    - head_rgb(features) → RGB delta (3 ch). Used for all tasks.
    - head_ab (features) → absolute Lab `ab` (2 ch). Overrides the ab
      channels of the RGB-delta output when config[0] (colorize axis)
      is active. Linear gate, fully differentiable, ONNX-friendly.

    Init:
      head_rgb: small Gaussian, std=0.01 — initial output ≈ input
                  (preserves identity-config behavior).
      head_ab : zero — initial ab = 0, so colorize=1 at step 0 produces
                  grayscale-from-input. Model learns to add color from
                  there. Mirrors v1 DDColor's training trajectory.
    """

    def __init__(self, in_dim: int):
        super().__init__()
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

        lab = rgb_to_lab(rgb_intermediate)
        w   = config[:, 0:1].view(-1, 1, 1, 1)             # colorize ∈ [0, 1]
        new_ab = w * ab_pred + (1.0 - w) * lab[:, 1:3]     # linear gate
        lab_out = torch.cat([lab[:, 0:1], new_ab], dim=1)
        return lab_to_rgb(lab_out)
```

### 3.1 Math properties

- `config[0] = 0`: `new_ab = lab[:, 1:3]` exactly → `output = rgb_intermediate`.
  `head_ab` receives zero gradient on this sample — clean separation.
- `config[0] = 1`: `new_ab = ab_pred`. Model only needs to predict ab;
  L is carried by `head_rgb`'s contribution (which also handles
  whatever other axes are active in the same sample).
- Compound `config[0]=1, config[1]=1, ...`: `head_rgb` denoises etc.
  AND fixes L; `head_ab` overrides only color. Each head does its
  specialty.

### 3.2 NAFNet vs PromptIR integration

The two backbones are slightly different in their existing output
arrangement:

| | NAFNet (current) | PromptIR (current) |
|---|---|---|
| Backbone-internal space | Lab | RGB |
| Head output | Lab delta (3 ch) | RGB delta (3 ch) |
| Final transform | `lab_to_rgb(lab_n + head(features))` | `rgb + head(features)` |

**NAFNet** changes:
- Replace `self.head: Conv2d(nf, 3)` with two heads:
  - `self.head_lab_delta: Conv2d(nf, 3)` — Lab delta (existing behavior)
  - `self.head_ab_abs:    Conv2d(nf, 2)` — absolute Lab ab (new)
- Forward composes:
  ```python
  lab_intermediate = lab_n + self.head_lab_delta(x)
  ab_pred         = self.head_ab_abs(x)
  w               = config[:, 0:1].view(-1, 1, 1, 1)
  ab_out          = w * ab_pred + (1.0 - w) * lab_intermediate[:, 1:3]
  L_out           = lab_intermediate[:, 0:1]
  return self.lab_to_rgb(torch.cat([L_out, ab_out], dim=1))
  ```
- NAFNet is *already* Lab-native, so this is the cleanest integration —
  no extra RGB↔Lab conversion vs the current code.

**PromptIR** changes:
- Replace `self.head: Conv2d(dim, 3)` with `self.dual_head: DualOutputHead(dim)`.
- Forward becomes:
  ```python
  return self.dual_head(features=d, rgb_input=rgb, config=config)
  ```
- PromptIR gains one extra `rgb_to_lab` + `lab_to_rgb` pair per forward
  (cheap; negligible vs the transformer stack).

The shared `DualOutputHead` class lives in `src/refine/models/heads.py`
and is used by PromptIR directly. NAFNet inlines the equivalent logic
because its intermediate is Lab, not RGB (different composition).

## 4. ONNX export wrapper

A new module `src/refine/export/wrapper.py`:

```python
class ONNXExportWrapper(nn.Module):
    """Stable ONNX entry point. Pins the export contract:
        forward(input, config) -> output

    where:
        input  : float32[B, 3, H, W] in [0, 1] (sRGB)
        config : float32[B, num_axes] in [0, 1] (axis activation flags)
        output : float32[B, 3, H, W] in [0, 1] (optionally clamped)

    Wraps any backbone whose Python forward is `(rgb, config) → rgb`.
    Decouples the exported graph signature from the model's internal
    forward signature, so future model changes can't break downstream
    consumers (C#, ORT-Web, TensorRT) as long as this wrapper holds.
    """

    def __init__(self, model: nn.Module, *, clamp_output: bool = False):
        super().__init__()
        self.model = model
        self.clamp_output = clamp_output

    def forward(self, input: torch.Tensor,
                config: torch.Tensor) -> torch.Tensor:
        out = self.model(input, config)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out
```

`export_onnx_from_model` is modified to wrap `model` in
`ONNXExportWrapper(model)` before calling `torch.onnx.export`. The
input/output names (`"input"`, `"config"`, `"output"`) and shapes are
unchanged. Parity verification still runs against the *unwrapped*
model so any wrapper-side bug shows up as a parity failure.

This is a defensive measure: today the wrapper is a pure pass-through,
but it gives us one place to add future contract-stable preprocessing
(e.g. input normalization, output clamping, dtype coercion) without
touching the backbone models.

## 5. Axis-probability rebalance

Production configs (`configs/laion-compound.yaml`,
`configs/promptir-laion.yaml`, `configs/nafnet-tiny-vivid.yaml`,
`configs/default.yaml`) get updated `axis_probs`:

```yaml
compound:
  axis_probs:
    colorize: 0.75   # was 0.5 — hard task, more exposure
    denoise:  0.40   # was 0.5 — easy task, already great
    sharpen:  0.75   # was 0.5 — hard task, more exposure
    dejpeg:   0.40   # was 0.5
    deblur:   0.40   # was 0.5
```

Expected: avg 2.7 axes active per non-identity sample (was 2.5).
Colorize/sharpen now in ~71% of non-identity samples (was 47%). Easy
tasks at ~38% — still plenty given they're working well.

The `CompoundConfig.axis_probs` schema is already a dict so this is
purely a config-value change. No code changes required.

## 6. Backward compatibility

### 6.1 Existing checkpoints

Old single-head checkpoints have parameters like `head.weight` and
`head.bias` but not the new dual-head names. We accept this gracefully:

- New models init `head_ab_abs` (NAFNet) or `dual_head.head_ab`
  (PromptIR) to zero, so a freshly-built model with no loaded weights
  behaves identically to today's single-head output (RGB delta only,
  zero ab override → no colorize-axis behavior change relative to a
  random init).
- `load_checkpoint` is updated to load with `strict=False` and warn
  about missing/unexpected keys, with a one-line table summarizing
  what was carried over and what was init-from-scratch.
- Renamed keys (`head.weight` → `head_lab_delta.weight` for NAFNet,
  `head.weight` → `dual_head.head_rgb.weight` for PromptIR) are
  remapped at load time via a small `_rename_legacy_keys` helper.

### 6.2 Existing ONNX files

Already-exported ONNX files are unaffected — they have the same I/O
contract and inference code doesn't change. New ONNX files exported
after this change will also have the same contract (now via the
wrapper).

## 7. Files added / modified

| File | Status | Purpose |
|---|---|---|
| `src/refine/models/heads.py` | NEW | `DualOutputHead` class (used by PromptIR; NAFNet inlines equivalent logic) |
| `src/refine/models/nafnet.py` | MOD | Replace `self.head` with `head_lab_delta` + `head_ab_abs`; update forward |
| `src/refine/models/promptir.py` | MOD | Replace `self.head` + final residual with `self.dual_head: DualOutputHead` |
| `src/refine/export/wrapper.py` | NEW | `ONNXExportWrapper` |
| `src/refine/export/onnx.py` | MOD | Wrap model in `ONNXExportWrapper` before `torch.onnx.export` |
| `src/refine/train/checkpoint.py` | MOD | `_rename_legacy_keys` helper + `strict=False` warning |
| `configs/default.yaml` | MOD | New `axis_probs` |
| `configs/laion-compound.yaml` | MOD | New `axis_probs` |
| `configs/promptir-laion.yaml` | MOD | New `axis_probs` |
| `configs/nafnet-tiny-vivid.yaml` | MOD | New `axis_probs` |
| `tests/test_dual_head.py` | NEW | shape, gate semantics (config=0 produces passthrough; config=1 produces lab+predicted_ab), gradient masking |
| `tests/test_onnx_wrapper.py` | NEW | Wrapper produces identical output to underlying model |
| `tests/test_nafnet.py` | MOD | Updated for two heads; existing param-count band adjusted |
| `tests/test_promptir.py` | MOD | Updated for two heads; existing param-count band adjusted |
| `tests/test_legacy_checkpoint_load.py` | NEW | Old single-head ckpt loads with strict=False; new params are zero-init |

## 8. Tests

| Test | Speed | What it proves |
|---|---|---|
| `test_dual_head::test_passthrough_when_colorize_zero` | fast | output equals `rgb + head_rgb(features)` *exactly* when `config[:,0]=0`; `head_ab` grads are zero |
| `test_dual_head::test_ab_override_when_colorize_one` | fast | output's Lab `ab` channels equal `head_ab(features)` when `config[:,0]=1` |
| `test_dual_head::test_linear_gate_continuous` | fast | At `config[:,0]=0.5`, output's `ab` is `0.5 * head_ab + 0.5 * passthrough_ab` |
| `test_onnx_wrapper::test_wrapper_is_pure_passthrough` | fast | `ONNXExportWrapper(model)(x, c)` is bitwise-equal to `model(x, c)` when `clamp_output=False` |
| `test_onnx_wrapper::test_clamp_applied` | fast | With `clamp_output=True`, output is in [0, 1] even when underlying produces overshoot |
| `test_promptir::test_onnx_export_parity_all_configs` | slow | Already exists; verifies the dual-head model exports cleanly to ONNX with parity |
| `test_nafnet::test_onnx_export_parity_all_configs` | slow | New; same verification for NAFNet |
| `test_legacy_checkpoint_load::test_old_ckpt_loads_with_warning` | fast | Single-head checkpoint loads into dual-head model via `_rename_legacy_keys`; new params zero-inited |
| `test_promptir_e2e_smoke` | slow | Updated to verify dual-head end-to-end: 10-step train with axis_probs rebalance produces non-zero ab predictions only on colorize samples |

## 9. Numerical / training contract

- Output range: same as before — clamping/clipping is the caller's
  responsibility (or the optional `clamp_output=True` on the export
  wrapper for inference-time safety).
- Gradient flow: `head_ab` gets zero gradient on samples where
  `config[:, 0] = 0`. This is *desirable* — it means the ab head only
  trains on colorize samples and isn't polluted by non-colorize
  gradients.
- AMP compatibility: the Lab↔RGB transforms are fp32-dispatched
  internally (existing behavior); the extra conversion in
  `DualOutputHead` follows the same path.

## 10. Loss behavior

No loss code changes. The implications:

- `l1_rgb` continues to operate on the final RGB output. It will now
  *partially* train both heads on colorize samples (the L portion via
  `head_rgb`, the ab portion via `head_ab`).
- `chroma_lab` continues to operate on the final RGB output's `ab`
  after rgb_to_lab. Its gradient now flows directly to `head_ab` on
  colorize samples — a much cleaner signal than today's setup where
  it has to fight through the entangled RGB delta.
- `colorfulness` operates on RGB output, unchanged behavior.
- `perceptual_vgg16bn` operates on RGB output, unchanged behavior.
- `freq_l1` operates on RGB output, unchanged behavior.

After this change, the loss preset weights from `nafnet-tiny-vivid`
that *didn't* work (chroma_lab=0.25, colorfulness=0.10) may work
*because* the structural impedance mismatch is gone. We'll re-run the
same recipe after landing the architecture change to test this
hypothesis.

## 11. Out of scope (deferred)

- GAN reintroduction — wait until colorize is structurally fixed.
- Curriculum learning — likely unnecessary if architecture fix works.
- Real-ESRGAN-style two-stage degradation for sharpen — separate
  improvement, evaluated after the dual-head experiment lands.
- CosAE HCM head — explicitly out of scope per 2026-05-14 discussion.
- Per-axis sub-decoders (separate decoder per task, shared encoder) —
  considered but rejected as a much heavier change with unclear
  marginal benefit over the dual-head approach.

## 12. Open questions

None blocking. The single design ambiguity — whether to use a hard
gate (`config[0] >= 0.5`) or a soft gate (linear mix) — was resolved
in favor of the soft gate because:
(a) it's fully differentiable,
(b) it's ONNX-export-clean (no `.float()` on a boolean),
(c) at training the values are always {0, 1} so it behaves like a
    hard gate anyway,
(d) at inference users can pass fractional values for partial-color
    effects, which is a nice bonus rather than a required feature.

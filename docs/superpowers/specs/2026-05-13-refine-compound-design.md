# Refine Compound — Conditioning-Vector Multi-Task Architecture

**Status:** Approved
**Date:** 2026-05-13
**Owner:** bglueck
**Lineage:** Successor to the integer-task-id NAFNet variant from the
2026-05-13-refine-multitask-design spec. Replaces (not extends) the
existing `nafnet` model.

## 1. Goal

Train one network that handles arbitrary combinations of restoration
axes (colorization, denoising, sharpening, JPEG-restore, deblur) in a
*single forward pass* per image. Caller specifies which axes to restore
via a 5-element conditioning vector; the model auto-detects degradation
*strength* from the input itself.

For video restoration in particular, this means each frame is one
forward pass — no chained inference, no per-task model selection.

## 2. Non-goals

- Keeping backward compatibility with v2 integer-task-id checkpoints
  (a complete replacement; old checkpoints become legacy).
- Adding a "blind / auto-detect-task" mode (caller still passes the
  vector saying *which* axes to address; the strength is auto-detected
  from the input). Could be added later as a small classifier.
- Continuous fractional config values during training (we train with
  binary {0, 1}). At inference the model will accept floats but
  behavior between 0 and 1 is undefined and may produce artifacts.

## 3. The 5 axes

| Index | Axis name | When `config[i]=1.0` | Degradation applied during training |
|---|---|---|---|
| 0 | `colorize` | restore color from grayscale | gray-out via LAB-L derivation |
| 1 | `denoise` | remove noise | add Gaussian noise σ ∈ [0.005, 0.05] (+ optional Poisson) |
| 2 | `sharpen` | sharpen / SR refine | bicubic downsample by random factor {2, 4, 8}, then bicubic upsample |
| 3 | `dejpeg` | remove JPEG artifacts | JPEG encode at random quality ∈ [20, 70], decode |
| 4 | `deblur` | remove blur | Gaussian blur σ ∈ [1.0, 3.0] (occasional motion blur) |

## 4. Architecture diff vs v2

Only the conditioning input changes. Every other module (NAFBlock,
TransformerBlock, color modules, head, residual) stays bit-identical.

```diff
- class TaskEmbed(nn.Module):
-     def __init__(self, *, num_tasks: int, dim: int = 128):
-         self.embed = nn.Embedding(num_tasks, dim)
-         self.mlp   = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
-     def forward(self, task: Tensor[B, long]) -> Tensor[B, dim]: ...

+ class ConfigEmbed(nn.Module):
+     def __init__(self, *, num_axes: int = 5, dim: int = 128):
+         self.proj  = nn.Linear(num_axes, dim)
+         self.mlp   = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))
+     def forward(self, config: Tensor[B, num_axes]) -> Tensor[B, dim]: ...
```

Model forward signature:

```python
def forward(self, rgb: Tensor[B, 3, H, W], config: Tensor[B, 5]) -> Tensor[B, 3, H, W]:
```

Parameter count change: `nn.Embedding(6, 128)` (768 params) → `nn.Linear(5, 128)` (640 + 128 bias = 768 params). Identical count to the v2 6-task version; smaller than the previous-6-tasks-and-growing path.

## 5. Training distribution

`CompoundDegradationWrapper` replaces `MultiTaskWrapper`:

For each sample:
1. With probability `identity_prob` (default 0.05): config = `[0, 0, 0, 0, 0]`, no degradation applied — model must output ≈ input.
2. Otherwise: roll 5 independent Bernoulli flags with per-axis probabilities (`axis_probs.colorize`, etc.; default 0.5 each).
3. Apply enabled degradations in real-world causal order (matches how real images get degraded):
   ```
   clean → deblur → denoise → sharpen → dejpeg → colorize → degraded
   ```
4. Emit:
   ```
   { "clean":     Tensor[3,H,W],
     "degraded":  Tensor[3,H,W],
     "config":    Tensor[5] float,
     "axes":      str ("color+denoise+sharp"; for logging only) }
   ```

Per-sample independent sampling means each batch covers a wide configuration mix. Empirically the model learns that the 5-bit configuration space is hyper-rectangular, so it generalizes to all 2⁵ corner configs even though no two batches have identical compositions.

## 6. Loss filter semantics

The existing `apply_to_tasks` per-loss field becomes `apply_to_axes`:

```yaml
losses:
  - name: l1_rgb
    weight: 1.0   # applies to every sample
  - name: colorfulness
    weight: 0.3
    apply_to_axes: [colorize]   # only on samples where colorize axis is 1
  - name: freq_l1
    weight: 0.2
    apply_to_axes: [sharpen, deblur]   # samples where ANY of these is 1
```

Filter semantics: "ANY of the listed axes is active for this sample." `LossSet` builds a sub-context with only the qualifying samples; if none qualify (empty filter result), the loss contributes 0.

## 7. Inference

```python
pipe = load_pipeline(checkpoint, device=...)
out = pipe.process(img_bgr, config={"colorize": True, "denoise": True, "sharpen": True})
# OR equivalently:
out = pipe.process(img_bgr, config=[1.0, 1.0, 1.0, 0.0, 0.0])
```

Pipeline pads input H/W to multiple of 16, runs one forward, unpads.

## 8. CLI

```bash
refine train  --config configs/laion-compound.yaml --data ~/data/laion-images
refine infer  --model <ckpt> --in frame.jpg --out fixed.jpg \
              --color --denoise --sharp [--dejpeg] [--deblur]
refine export --model <ckpt> --output model.onnx [--dynamic-hw]
refine info   --model <ckpt>          # shows model_type, axes, sidecar info
```

Removed: `list-tasks` (no enumerable task list anymore — there are always 5 axes).

Per-flag CLI: each axis is a typed `--axis/--no-axis` option, default false. Self-documenting via `--help`.

## 9. ONNX export

Inputs:
| Name | Type | Shape | Range |
|---|---|---|---|
| `input` | float32 | (B, 3, H, W) or dynamic | [0, 1] RGB |
| `config` | float32 | (B, 5) | [0, 1] per axis |
| `output` | float32 | (B, 3, H, W) | [0, 1] RGB |

Per-config parity verification at export time: 7 reference configs are checked against PyTorch within `1e-3`: identity `[0,0,0,0,0]`, all-on `[1,1,1,1,1]`, and each of the 5 single-axis-on configs. Catches silent config-baking by the exporter.

## 10. Sidecar JSON

Written alongside both `final.pt` and any exported `.onnx`:

```json
{
  "model_type": "nafnet",
  "model_size": "large",
  "input_size": 256,
  "config_axes": ["colorize", "denoise", "sharpen", "dejpeg", "deblur"],
  "version": "0.2.0"
}
```

Downstream consumers (C# integration, ORT-Web users) can read this once and know the contract without parsing a PyTorch checkpoint.

## 11. Configs

`degradations:` section restructured:

```yaml
compound:
  identity_prob: 0.05
  axis_probs:
    colorize: 0.5
    denoise:  0.5
    sharpen:  0.5
    dejpeg:   0.5
    deblur:   0.5
  # Per-axis degradation parameters (identical to v2 base degradations):
  colorize: {}
  denoise:  { sigma_range: [0.005, 0.05], poisson_prob: 0.0 }
  sharpen:  { factor_choices: [2, 4, 8] }
  dejpeg:   { quality_range: [20, 70] }
  deblur:   { sigma_range: [1.0, 3.0], motion_prob: 0.2 }
```

Replaces the v2 `degradations: {name: {weight, ...}}` dict. The `sharpen` axis is the only one whose params differ from v2 (added `factor_choices` — the model trains with random SR factor per sample).

## 12. Files modified

| File | What changes |
|---|---|
| `src/refine/config.py` | New `CompoundConfig` model; replaces `degradations: dict` |
| `src/refine/models/task_embed.py` | Rename → `ConfigEmbed`, replace `nn.Embedding` with `nn.Linear` |
| `src/refine/models/nafnet.py` | Forward signature `(rgb, config)`; `build_model` no longer takes `num_tasks` |
| `src/refine/models/registry.py` | `build_model(cfg, num_axes=5)` instead of `num_tasks` |
| `src/refine/data/compound.py` | NEW — `CompoundDegradationWrapper` |
| `src/refine/data/multitask.py` | DELETE |
| `src/refine/data/degradations/superres.py` | Add a `SharpenSR` that handles random factor in {2,4,8} per call; `sr_x2`/`sr_x4` removed |
| `src/refine/losses/__init__.py` | `apply_to_axes` (replaces `apply_to_tasks`) |
| `src/refine/losses/registry.py` | `LossContext.task_names` → `axes_active` (list[str]) and `LossContext.task_ids` removed (no longer needed for indexing) |
| `src/refine/train/trainer.py` | Use `CompoundDegradationWrapper`; per-axis PSNR; preview shows 6 axes rows |
| `src/refine/train/preview.py` | Caption format: 6 fixed task rows ("identity" + each single axis) + "all-on" |
| `src/refine/train/ui.py` | Per-axis PSNR panel rows |
| `src/refine/infer/pipeline.py` | Take `config: dict | list | Tensor`, pad to multiple of 16 |
| `src/refine/cli.py` | Per-flag axis options; `list-tasks` → `info` |
| `src/refine/export/onnx.py` | 2-input float ONNX; per-config parity check (7 configs) |
| `configs/default.yaml`, `tiny.yaml`, `large.yaml`, `laion-multitask.yaml` | Restructure to new `compound:` section; rename `laion-multitask.yaml` → `laion-compound.yaml` |
| `tests/*` | Adjust every test that touched task IDs |

## 13. Non-goals (deferred)

- Continuous fractional configs during training (only binary).
- Auto-detect-task mode (no extra classifier head).
- Per-axis training-time strength curricula (random uniform within configured range).
- Real-world multi-stage SR degradation (Real-ESRGAN-style two-stage; future work).

## 14. Open questions

None blocking.

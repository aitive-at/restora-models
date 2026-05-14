# Refine — Multi-Task Image Restoration

**Status:** Approved
**Date:** 2026-05-13
**Owner:** bglueck
**Target hardware:** NVIDIA Blackwell, 96 GB VRAM, single-GPU
**Lineage:** Successor to coliraz (the v1 colorization-only project, archived under `legacy/coliraz-v1/`).

## 1. Goal

A single PyTorch codebase, managed with `uv`, that trains one unified neural
network on multiple low-level image restoration tasks simultaneously. The
infrastructure (trainer loop, AMP, EMA, Rich live UI, periodic comparison
PNG previews, modular loss registry, ONNX export with parity verification,
test suite) is lifted from coliraz v1; the architecture, dataset pipeline,
and CLI are rewritten to handle:

- multiple tasks in any combination
- multiple super-resolution target factors
- RGB-in / RGB-out at the same resolution (color conversions baked into
  the model where used, so the exported ONNX is a single self-contained graph)

## 2. Non-goals

- Distributed / multi-GPU training (single Blackwell only).
- Video inference.
- Live web demo.
- Reference scaling beyond what the SR task already produces.
- Conversion of v1 coliraz checkpoints (different architecture, different
  output channel count, different input contract).
- Automatic ("blind") degradation detection at inference — the caller
  passes `--task <name>`. A blind-mode head can be added later as an
  extension.

## 3. Strategy

NAFNet-based U-Net (the cleanest SOTA single-task baseline available)
extended for multi-task with three additions:

- **Lightweight task conditioning via FiLM (conv blocks) and AdaLN
  (transformer bottleneck)** — `nn.Embedding(num_tasks, D)` → MLP → per-block
  modulation scales (γ, β). ~3 % extra parameters.
- **Transformer bottleneck** at the lowest-resolution feature map for the
  global context that colorization specifically needs.
- **Frozen RGB ↔ LAB conversion layers** at the model's input and output,
  with the network operating in normalized LAB space internally. Residual
  learning: `out_lab_n = in_lab_n + delta_lab_n`, so an untrained model
  produces input-passes-through (gray-input → gray-output for colorization,
  no spurious refinement for SR/denoising, etc.).

A `register_model` decorator and a small backbone interface (`forward(rgb,
task) -> rgb`) make it straightforward to plug in alternative architectures
later (Restormer, PromptIR, MambaIR, custom) without touching the
training, data, or export code.

## 4. Project structure

```
coliraz/                                # the existing git repo, renamed in spirit but kept
├── legacy/coliraz-v1/                  # ← git mv of every v1 file
│   ├── src/, configs/, tests/, docs/, pyproject.toml, main.py, uv.lock
│   └── README.md (note: "archived v1; new project at ../../")
├── reference/ddcolor_original/         # stays
├── runs/                               # stays — user training outputs
├── docs/
│   ├── integration/                    # stays (ONNX C# guide still useful, RGB-relevant)
│   └── superpowers/specs/              # this doc lives here
├── src/refine/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models/
│   │   ├── registry.py
│   │   ├── color.py                    # RgbToLab + LabToRgb nn.Modules (fp32 always)
│   │   ├── task_embed.py               # nn.Embedding + MLP for task conditioning
│   │   ├── nafblock.py                 # NAFBlock with optional FiLM
│   │   ├── transformer_block.py        # MHSA + FFN with AdaLN
│   │   └── nafnet.py                   # the registered backbone
│   ├── losses/
│   │   ├── registry.py                 # ColorizationLoss base + LossContext (renamed)
│   │   ├── pixel.py                    # l1_rgb, l2_rgb, charbonnier_rgb
│   │   ├── perceptual.py               # VGG16-BN (already RGB-native)
│   │   ├── gan.py                      # GAN + UNet discriminator
│   │   ├── colorfulness.py             # ported, applied via apply_to_tasks
│   │   ├── freq_l1.py                  # FFT-magnitude L1 (new, for SR/deblur)
│   │   ├── lpips.py                    # optional perceptual (new)
│   │   ├── metrics.py                  # psnr_metric, ssim_metric (no-grad logging only)
│   │   └── __init__.py                 # LossSet composer + apply_to_tasks masking
│   ├── data/
│   │   ├── dataset.py                  # RecursiveImageDataset (clean RGB only)
│   │   ├── transforms.py
│   │   ├── multitask.py                # MultiTaskWrapper: per-sample task picker
│   │   └── degradations/
│   │       ├── registry.py             # @register_degradation + Degradation base
│   │       ├── colorization.py
│   │       ├── denoise.py
│   │       ├── superres.py             # sr_x2 and sr_x4 from one file (factor param)
│   │       ├── deblur.py
│   │       └── jpeg.py
│   ├── train/
│   │   ├── trainer.py                  # 95 % lifted from v1; multi-task batch path
│   │   ├── ui.py                       # lifted; per-task PSNR rows
│   │   ├── preview.py                  # multi-task grid (1 row per task)
│   │   ├── ema.py                      # lifted verbatim
│   │   ├── checkpoint.py               # lifted; writes <ckpt>.task_map.json sidecar
│   │   └── auto_batch.py
│   ├── infer/pipeline.py               # MultiTaskRefinerPipeline (rgb + task → rgb)
│   ├── export/onnx.py                  # 2-input ONNX, per-task parity check
│   └── utils/{color.py, gpu.py, timing.py, io.py}
├── configs/
│   ├── default.yaml
│   ├── tiny.yaml
│   ├── large.yaml
│   └── laion-multitask.yaml
├── tests/                              # ~16 tests, all CPU, < 30 s total
├── pyproject.toml                      # name = "refine"; scripts: refine = restora_models.cli:app
├── main.py
└── README.md
```

### Archival mechanics

Single migration commit:

```
mkdir -p legacy/coliraz-v1
git mv src configs tests pyproject.toml main.py uv.lock legacy/coliraz-v1/
# docs/ stays at repo root (sub-docs in docs/integration/ are model-agnostic)
# runs/ stays at repo root (user training outputs)
```

`legacy/coliraz-v1/` keeps its own `pyproject.toml`. To resurrect:

```
cd legacy/coliraz-v1 && uv sync && uv run coliraz train ...
```

## 5. Model architecture

**Single forward path** (shapes for 256 × 256, `nf=64`, task embedding `D=128`):

```
Inputs:
  rgb      (B, 3, H, W)  float32, [0, 1]
  task     (B,)          int64,   task ID

1.  RgbToLab (frozen, fp32 always)
       RGB [0,1] → LAB cv2 convention → normalized lab_n
       (L_n = (L-50)/50, a_n = a/110, b_n = b/110)
       → (B, 3, H, W) lab_n

2.  TaskEmbed
       nn.Embedding(num_tasks, D=128)
       → MLP(D → D → D)
       → task_vec (B, 128)

3.  Stem: Conv2d 3 → nf, k=3      → feat0 (B, nf, H, W)

4.  Encoder (4 downsamples → 5 resolution levels)
       After stem:   level /1  with nf      channels
       Stage 0: NAFBlock × N_0 at level /1, then down-2x → level /2
       Stage 1: NAFBlock × N_1 at level /2, then down-2x → level /4
       Stage 2: NAFBlock × N_2 at level /4, then down-2x → level /8
       Stage 3: NAFBlock × N_3 at level /8, then down-2x → level /16
       Channels per level: nf, 2·nf, 4·nf, 8·nf, 16·nf
       Each NAFBlock: FiLM conditioning from task_vec

5.  Bottleneck  (B, 16·nf, H/16, W/16)
       1×1 projection 16·nf → hidden_dim (default 384)
       N_bottle × transformer block: LN → MHSA (SDPA) → res → LN → FFN → res
       AdaLN modulation per block: γ, β = MLP(task_vec)
       1×1 projection hidden_dim → 16·nf
       At H=W=256, /16 = 16² = 256 tokens — cheap with FlashAttention

6.  Decoder (mirrors encoder, 4 upsamples)
       Stage 3: pixel-shuffle up-2x at /16 + skip-concat with enc /8 + NAFBlock × N_3
       Stage 2: same pattern, /4
       Stage 1: same pattern, /2
       Stage 0: same pattern, /1
       FiLM conditioning at every NAFBlock
       ends at (B, nf, H, W)

7.  Head: Conv2d nf → 3, k=3  → delta_lab_n (B, 3, H, W)

8.  Residual: out_lab_n = lab_n + delta_lab_n
       Denormalize: L = 50 + 50·L_n; a = 110·a_n; b = 110·b_n

9.  LabToRgb (frozen, fp32 always) → clamp [0, 1]
       → output (B, 3, H, W) RGB
```

### Component details

**NAFBlock** (NAFNet, Chen et al. ECCV'22):

```
x → LayerNorm
  → Conv 1x1 (expand 2c)
  → DWConv 3x3
  → SimpleGate (split half-half, multiply)
  → Channel attention (squeeze-excite-lite)
  → Conv 1x1 (back to c)
  → DropPath → residual
  → LayerNorm
  → FFN (Conv 1x1 4c → SimpleGate → Conv 1x1 c)
  → DropPath → residual
```

**FiLM conditioning** (conv blocks):

```
γ, β = Linear(task_vec, 2·c)
x = γ * x + β   (applied after the first LayerNorm in each NAFBlock)
```

**AdaLN modulation** (bottleneck transformer, adapted from DiT):

```
γ1, β1, γ2, β2 = Linear(task_vec, 4·c)
x = γ1 · LN(x) + β1   then attention   then γ2 · LN(x) + β2   then FFN
```

**LAB conversion as `nn.Module`:**

```python
class RgbToLab(nn.Module):
    def forward(self, rgb):
        with torch.amp.autocast("cuda", enabled=False):
            return self._normalize(rgb_to_lab(rgb.float()))

class LabToRgb(nn.Module):
    def forward(self, lab_n):
        with torch.amp.autocast("cuda", enabled=False):
            return lab_to_rgb(self._denormalize(lab_n)).clamp(0, 1)
```

Frozen (no learned params). Always runs in fp32 — bf16 numerical issues with
`pow(x, 1/2.4)` were the dominant failure mode in v1 and are now an
architectural property of the conversion layers themselves.

### Two preset sizes

| Preset | `nf` | enc_depths | bottle_blocks | hidden_dim | params | use case |
|---|---|---|---|---|---|---|
| `tiny` | 32 | [2, 2, 2, 2] | 2 | 256 | ~17 M | fast iteration; mobile-class |
| `large` | 64 | [2, 2, 4, 8] | 4 | 384 | ~64 M | full quality |

### Model registry

```python
@register_model("nafnet")
class NAFNetMultiTask(nn.Module):
    def __init__(self, cfg: ModelConfig, num_tasks: int): ...
    def forward(self, rgb: Tensor, task: Tensor) -> Tensor: ...
```

`config.model.type: nafnet` selects the backbone. Future alternatives
(`restormer`, `promptir`, `mambair`) register the same way and slot in
without changes to data/training/export code.

## 6. Degradation pipeline

### Six tasks, one file each (sr_x2 and sr_x4 share `superres.py` via the `factor` param)

| Task | name | task_id (default) | What it does |
|---|---|---|---|
| Colorization | `colorize` | 0 | RGB → LAB; a=b=0; LAB → RGB (gray-as-RGB) |
| Denoising | `denoise` | 1 | Add Gaussian noise σ ∈ [0.005, 0.05]; optional Poisson+read mix |
| Super-resolution × 2 | `sr_x2` | 2 | Bicubic downsample → bicubic upsample (factor 2) |
| Super-resolution × 4 | `sr_x4` | 3 | Bicubic downsample → bicubic upsample (factor 4) |
| Deblur | `deblur` | 4 | Gaussian blur σ ∈ [1.0, 3.0]; optional motion blur |
| JPEG-restore | `jpeg` | 5 | Encode at JPEG quality ∈ [20, 70], decode |

Each degradation is one `Degradation` subclass exposing
`degrade(rgb: np.ndarray, rng: random.Random) -> np.ndarray`. All produce
same-resolution RGB.

### YAML configuration

```yaml
degradations:
  colorize: { weight: 1.0 }
  denoise:  { weight: 1.0, sigma_range: [0.005, 0.05] }
  sr_x2:    { weight: 1.0, factor: 2 }
  sr_x4:    { weight: 1.0, factor: 4 }
  deblur:   { weight: 0.7, sigma_range: [1.0, 3.0], motion_prob: 0.2 }
  jpeg:     { weight: 0.7, quality_range: [20, 70] }
```

Comment out a line to skip the task. Task IDs are assigned in YAML
declaration order at config-load time.

### Per-sample task picker

```python
class MultiTaskWrapper(Dataset):
    def __init__(self, clean_ds, degradations, weights, seed): ...
    def __getitem__(self, idx) -> dict:
        rgb = self.clean[idx]                # (3, H, W) tensor [0, 1]
        rng = random.Random((seed * 1_000_003) ^ idx)
        u = rng.random()
        task_idx = int(np.searchsorted(self.cdf, u))
        deg = self.degs[task_idx]
        degraded = deg.degrade(rgb.permute(1, 2, 0).numpy(), rng)
        return {
            "clean":     rgb,
            "degraded":  torch.from_numpy(degraded.transpose(2, 0, 1)),
            "task_id":   torch.tensor(deg.task_id, dtype=torch.long),
            "task_name": deg.name,
        }
```

Each batch contains a heterogeneous mix of tasks. The model handles each
sample through its task embedding.

### Clean dataset

`RecursiveImageDataset` is lifted from v1: recursive scan, manifest cache
(`.refine-manifest.txt`), PIL header-only size filter, deterministic
val/train split by path hash, random crop + hflip. Returns clean
`(3, H, W)` float32 RGB only — degradation lives outside.

## 7. Loss system

Same registry pattern as v1, generalized to RGB.

### `LossContext` (extended for multi-task)

```python
@dataclass
class LossContext:
    pred_rgb:     Tensor   # (B, 3, H, W) model output
    clean_rgb:    Tensor   # (B, 3, H, W) GT
    degraded_rgb: Tensor   # (B, 3, H, W) model input (for residual-aware losses)
    task_ids:     Tensor   # (B,) int64
    task_names:   list[str]
    discriminator: nn.Module | None = None
```

### Ported / generalized losses

| Name | Operates on | Notes |
|---|---|---|
| `l1_rgb` | pred_rgb vs clean_rgb | basic pixel match |
| `charbonnier_rgb` | same | smoother L1 |
| `perceptual_vgg16bn` | RGB vs RGB | unchanged from v1 |
| `gan` | via discriminator | optional |
| `colorfulness` | pred_rgb | typically only colorize task |

### New losses

| Name | Notes |
|---|---|
| `freq_l1` | L1 in FFT magnitude; helps SR/deblur detail recovery |
| `lpips` | learned perceptual (AlexNet backbone); optional |
| `psnr_metric` | log-only, per-task; no gradient |
| `ssim_metric` | log-only, per-task; no gradient |

### Per-task masking

Every loss accepts an optional `apply_to_tasks` filter:

```yaml
losses:
  - { name: l1_rgb,             weight: 1.0 }
  - { name: perceptual_vgg16bn, weight: 0.5 }
  - { name: colorfulness,       weight: 0.3, apply_to_tasks: [colorize] }
  - { name: freq_l1,            weight: 0.2, apply_to_tasks: [sr_x2, sr_x4, deblur] }
  - { name: gan,                weight: 0.1, apply_to_tasks: [colorize, sr_x2, sr_x4] }
```

The composer computes each loss only over rows whose task is in the filter.
Empty mask → contribution 0 and reported as such.

### Loss presets

| Preset | Composition |
|---|---|
| `minimal` | l1_rgb @ 1.0 |
| `standard` | l1_rgb + perceptual + colorfulness(colorize) + freq_l1(sr_x2/sr_x4/deblur) |
| `vivid` | standard + colorfulness weight 2.0 + train.color_enhance: true |
| `full` | standard + gan(colorize/sr_x2/sr_x4) + lpips |

## 8. Trainer

95 % lifted from v1. Key concrete diff:

```diff
- pred_ab = self.model(gray_rgb)
- ctx = LossContext(pred_ab=, gt_ab=, pred_rgb=, gt_rgb=, gray_rgb=, ...)
+ pred_rgb = self.model(degraded, task_id)
+ ctx = LossContext(pred_rgb=, clean_rgb=, degraded_rgb=, task_ids=, task_names=, ...)
```

Inherited from v1, unchanged:
- AMP / bf16 + grad-norm guard (skips opt step on Inf gradients)
- 20-consecutive-NaN abort with recovery hint
- ModelEMA decay 0.999 (fp32 shadow)
- channels-last memory format
- cosine LR scheduler with warmup
- atomic checkpoint save/load
- per-step keypress handler (q quit, s save, p preview)
- Rich live UI shell

### Per-task metrics in UI

The losses panel surfaces per-task PSNR and per-loss values. Implementation:

- Per-row metric tensors (no-grad)
- Accumulate into per-task buckets
- EMA-smoothed for display

### Preview grid

One row per task, 4 columns (`clean | degraded | predicted | |Δ| heatmap`).
Fixed validation samples (deterministic by path hash) plus optional random
samples; same atomic-write + rotated history mechanism as v1.

## 9. Inference

`MultiTaskRefinerPipeline`:

```python
def process(self, img_bgr: np.ndarray, task: str) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_padded, pads = pad_to_multiple(rgb, multiple=16, mode="reflect")
    t = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
    task_id = torch.tensor([self.task_name_to_id[task]], dtype=torch.long, device=self.device)
    out = self.model(t, task_id).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = unpad(out, *pads)
    return (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255.0).round().clip(0, 255).astype(np.uint8)
```

- Caller responsible for pre-upsampling low-resolution images before
  calling with `task="sr_x4"`, matching the training distribution. CLI
  exposes a `--upsample-to WxH` convenience flag.
- 16-pixel padding handled by the pipeline, not the model. The model
  itself is stride-16 internally (4 encoder downsamples); arbitrary
  inputs get reflection-padded up to a multiple of 16 before the
  forward and unpadded afterward. ONNX consumers can either go through
  this pipeline or pre-pad in their own runtime.

## 10. ONNX export

```python
torch.onnx.export(
    model_eval,
    args=(dummy_rgb, dummy_task_id),
    input_names=["input", "task"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch", 2: "height", 3: "width"} if dynamic_hw else {0: "batch"},
        "task":   {0: "batch"},
        "output": {0: "batch", 2: "height", 3: "width"} if dynamic_hw else {0: "batch"},
    },
    opset_version=17,
)
```

| Name | Type | Shape | Range |
|---|---|---|---|
| `input` | `float32` | (B, 3, H, W) or dynamic h,w | [0, 1] RGB |
| `task` | `int64` | (B,) | [0, num_tasks) |
| `output` | `float32` | (B, 3, H, W) or dynamic h,w | [0, 1] RGB |

### Per-task parity verification

```python
for task_id in range(num_tasks):
    x_rgb = np.random.rand(1, 3, S, S).astype(np.float32)
    t = np.array([task_id], dtype=np.int64)
    ort_out = sess.run(None, {"input": x_rgb, "task": t})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x_rgb), torch.from_numpy(t)).numpy()
    assert max(abs(ort_out - torch_out)) < 1e-3
```

Catches silent shape/conditioning baking that affects only some tasks.

### Task-map sidecar

Written alongside both `final.pt` and any `.onnx`:

```json
{
  "tasks": { "colorize": 0, "denoise": 1, "sr_x2": 2, "sr_x4": 3, "deblur": 4, "jpeg": 5 },
  "input_size": 256,
  "model_size": "large",
  "version": "0.1.0"
}
```

So downstream consumers (e.g. the C# integration from v1's docs) can read
the task contract without parsing a PyTorch checkpoint.

## 11. CLI

```bash
refine train      --config configs/laion-multitask.yaml --data ~/data/laion-images
refine train      --config configs/large.yaml --resume runs/<name>/ckpt/last.pt --compile
refine infer      --model <ckpt> --input photo.jpg --output out.jpg --task colorize
refine infer      --model <ckpt> --input lowres.jpg --output sr.jpg --task sr_x4 --upsample-to 2048x2048
refine export     --model <ckpt> --output model.onnx --input-size 256 --dynamic-hw
refine scan-data  --root /path/to/images
refine list-tasks --config configs/laion-multitask.yaml
```

CLI overrides merge into the YAML config the same way as v1. `--task`
accepts task **names** (not integer IDs); CLI looks up the sidecar map.

## 12. Configs

```
configs/default.yaml            # full surface with sensible defaults
configs/tiny.yaml               # model.size: tiny, larger batch
configs/large.yaml              # model.size: large, smaller batch
configs/laion-multitask.yaml    # concrete preset for the LAION shard
```

Layered: `default.yaml` is the base, the others inherit via `defaults:`
and override only what changes.

## 13. Dependencies (uv-managed)

```toml
[project]
name = "refine"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.4", "torchvision>=0.19",
  "opencv-python-headless>=4.10",
  "numpy>=1.26,<3", "pillow>=10",
  "typer>=0.12", "pydantic>=2.7", "pyyaml>=6.0", "rich>=13.7",
  "onnx>=1.16", "onnxruntime>=1.19", "onnxsim>=0.4", "onnxscript>=0.1",
  "nvidia-ml-py>=12.0",
  "tqdm>=4.66",
]
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
[project.scripts]
refine = "restora_models.cli:app"
```

Note: `timm` is **not** in the dependency list (no more ConvNeXt encoder).
`lpips` is added only as an optional extra if/when LPIPS loss is enabled.

## 14. Testing strategy

CPU-only, full suite < 30 s, slow tests (full ONNX parity, e2e smoke) gated
by `REFINE_SLOW=1`.

| Test | Verifies |
|---|---|
| `test_color.py` | rgb↔lab round-trip; cv2 parity within ε |
| `test_rgb_to_lab_module.py` | the nn.Module conversion (autocast disabled in forward) |
| `test_degradations.py` | each degradation produces same-resolution RGB, deterministic with same rng seed |
| `test_multitask_dataset.py` | weighted task sampling, per-sample task variety in a batch |
| `test_task_embed.py` | nn.Embedding + MLP shape contract |
| `test_nafblock.py` | NAFBlock forward shape; FiLM modulation actually changes output |
| `test_transformer_block.py` | bottleneck transformer + AdaLN |
| `test_nafnet_forward.py` | full model `(rgb, task) → rgb`; output shape == input shape; residual identity at init |
| `test_losses_rgb.py` | l1_rgb, charbonnier_rgb, perceptual, colorfulness, freq_l1 each return scalar with grad |
| `test_loss_set_apply_to_tasks.py` | per-task masking computes correct subsets |
| `test_metrics.py` | psnr/ssim no-grad, per-task accumulation |
| `test_trainer_step.py` | one-step overfit with 2-task mix reduces loss |
| `test_preview_multitask.py` | grid renderer with N task rows produces a valid PNG |
| `test_inference.py` | pipeline rgb-to-rgb at non-32 sizes (reflect padding works) |
| `test_export_onnx.py` | both fixed and dynamic, per-task parity for all tasks |
| `test_cli.py` | --help works for each subcommand; scan-data runs end-to-end on a tiny fixture |

## 15. Migration sequence (high level)

The implementation plan in the next step will produce a step-by-step TDD
plan. Coarse order:

1. Archive: `git mv` v1 to `legacy/coliraz-v1/`, write new top-level README.
2. Scaffold new `src/refine/`, `pyproject.toml`, `tests/conftest.py`.
3. Utilities (color, gpu, timing) — mostly lifted, lightly renamed.
4. Color conversion `nn.Module`s with autocast-disabled forward.
5. Pydantic config + YAML loader (`!preset`, chained defaults).
6. Degradation registry + 5 task implementations.
7. Clean dataset (lifted) + MultiTaskWrapper.
8. NAFBlock, transformer block, task embed, model registry, NAFNet.
9. Loss registry + RGB-space losses + apply_to_tasks masking + metrics.
10. EMA + checkpoint + preview + UI (lifted, multi-task adaptations).
11. Trainer wiring everything together.
12. Inference pipeline + reflection padding.
13. ONNX export with 2 inputs + per-task parity check + task-map sidecar.
14. CLI (Typer): train, infer, export, scan-data, list-tasks.
15. End-to-end smoke test (synthetic data, 2 tasks).

## 16. Open questions

None blocking implementation. Deferred:

- Blind degradation classifier head (model auto-detects task at inference).
  Adds ~1 % params; can be tacked on as a separate module if desired later.
- Real-world noise simulation (`Real-ESRGAN`-style two-stage degradation
  for SR) for higher-fidelity training inputs. Can be added as a
  `sr_real` task variant.
- LPIPS dependency: pulled in optionally only when LPIPS loss is enabled,
  to keep the default install lean.

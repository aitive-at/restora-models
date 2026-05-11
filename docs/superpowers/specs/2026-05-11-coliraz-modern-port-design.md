# Coliraz — Modern PyTorch Port of DDColor

**Status:** Approved
**Date:** 2026-05-11
**Owner:** bglueck
**Target hardware:** NVIDIA Blackwell, 96 GB VRAM, single-GPU

## 1. Goal

Port `./DDColor/` (ICCV 2023 image colorization) to a clean, modern PyTorch 2.x codebase managed with `uv`. Provide:

- A small, focused source tree (no BasicSR framework).
- A `coliraz` CLI with `train`, `infer`, `export`, `scan-data` subcommands.
- Training that reads images recursively from a directory and synthesizes grayscale/color pairs on the fly.
- A live terminal dashboard (no TensorBoard/W&B) plus a sample comparison PNG refreshed every ~10 s.
- ONNX export with numerical-parity verification.
- All efficiency tricks reasonable on a Blackwell 96 GB card (bf16 AMP, channels-last, SDPA/FlashAttention, `torch.compile`, fused AdamW, channels-last, cuDNN benchmark).
- Full test suite that runs on CPU in under 30 s.

## 2. Non-goals

- Distributed training (DDP/FSDP) — single GPU only.
- Web/Gradio demo.
- Live FID/CF metrics during training.
- TensorBoard / W&B integration.
- Video colorization.
- Automatic conversion of the public DDColor checkpoints (separate one-off script if requested later).

## 3. Strategy

**Approach A — fresh rewrite with `timm` encoder, opting for clean modern code over weight-compatibility with public DDColor checkpoints.** The user is training on their own dataset, so binary compatibility with the original `.pth` files is not a constraint.

Key wins vs. the original:

- Replace `ConvNeXt + Hook` system with `timm.create_model("convnext_*", features_only=True)`. Default variants: `convnext_tiny.fb_in22k` (tiny) and `convnext_large.fb_in22k` (large). The variant name is overridable via `model.encoder_variant` for users who want a different pretrained source.
- Replace hand-rolled multi-head attention with `nn.MultiheadAttention(batch_first=True)` → dispatches to `F.scaled_dot_product_attention` → FlashAttention on Blackwell.
- Replace legacy `nn.utils.spectral_norm` with `nn.utils.parametrizations.spectral_norm` (compile-safe).
- Replace `torch.cuda.amp` with the new unified `torch.amp` API.
- Replace YAML-driven BasicSR registry/loader machinery with a small Pydantic config + a single trainer class.
- Replace ad-hoc `print`/`logging` with `rich.live` dashboard.
- Replace manual `pretrain/` weight downloads with `timm`'s HuggingFace cache.

## 4. Project structure

```
coliraz/
├── reference/ddcolor_original/        # moved from ./DDColor/ — read-only reference
├── src/coliraz/
│   ├── cli.py                         # Typer CLI: train | infer | export | scan-data
│   ├── config.py                      # Pydantic config models, YAML loader, !preset tag
│   ├── models/
│   │   ├── ddcolor.py                 # top-level DDColor module
│   │   ├── encoder.py                 # timm ConvNeXt feature extractor
│   │   ├── pixel_decoder.py           # UNet-style upsample path
│   │   ├── color_decoder.py           # transformer w/ color queries (SDPA)
│   │   ├── unet_blocks.py             # PixelShuffle-ICNR, UnetBlockWide
│   │   ├── refine.py                  # final 1×1 spectral-norm conv
│   │   └── discriminator.py           # UNet discriminator (optional)
│   ├── losses/
│   │   ├── registry.py                # name → loss factory; LossSet composer
│   │   ├── pixel.py                   # L1, L2, Charbonnier
│   │   ├── perceptual.py              # VGG16-bn perceptual + style
│   │   ├── gan.py                     # vanilla/lsgan/hinge + disc loss
│   │   └── colorfulness.py
│   ├── data/
│   │   ├── dataset.py                 # RecursiveImageDataset + manifest cache
│   │   ├── grayscale.py               # RGB → RGB-of-gray (via LAB-L)
│   │   └── transforms.py              # crop / flip / resize
│   ├── train/
│   │   ├── trainer.py                 # train loop, AMP, EMA, scheduler
│   │   ├── ui.py                      # Rich live dashboard
│   │   ├── preview.py                 # background thread: comparison grid PNG
│   │   ├── auto_batch.py              # optional batch-size autotuner
│   │   └── checkpoint.py              # save/load (best, last, periodic)
│   ├── infer/pipeline.py              # LAB pipeline (full-res L + resize AB)
│   ├── export/onnx.py                 # ONNX export + parity check
│   └── utils/
│       ├── color.py                   # vectorized rgb↔lab on tensors
│       ├── gpu.py                     # pynvml wrapper (optional)
│       └── timing.py
├── configs/
│   ├── default.yaml
│   ├── tiny.yaml
│   └── large.yaml
├── tests/                             # pytest, CPU-only, <30 s total
├── pyproject.toml                     # uv-managed, [project.scripts] coliraz = "coliraz.cli:app"
├── main.py                            # thin wrapper that calls coliraz.cli:app()
└── README.md
```

## 5. Model architecture

The user-visible forward contract is preserved: `(B, 3, H, W) RGB-of-gray → (B, 2, H, W) AB`.

```
RGB-of-gray
   │
   ▼  timm ConvNeXt (features_only, out_indices=(0,1,2,3))
[f0, f1, f2, f3]   multi-scale features
   │
   ▼  PixelDecoder (UnetBlockWide × 3 + PixelShuffle-ICNR ×4)
hi-res pixel features  +  [m0, m1, m2]   mid-scale features
   │                                │
   │                                ▼  ColorDecoder
   │                          - 100 learnable color queries
   │                          - 9 transformer layers
   │                          - cross-attn over m0..m2 with positional embeddings
   │                          - self-attn + FFN per layer
   │                          - SDPA backend (FlashAttention on Blackwell)
   │                                │
   │                                ▼  output Q×C color embeddings
   ▼                                │
einsum("bqc,bchw->bqhw")            │
   │                                │
   ▼                                │
Q-channel coarse map  ◄─────────────┘
   │
   ▼  concat(coarse, input_rgb) → (Q+3) channels
   ▼  Refine 1×1 conv (parametrizations.spectral_norm)
AB output (B, 2, H, W)
```

Config knobs:

```yaml
model:
  size: tiny | large
  input_size: 256 | 384 | 512
  num_queries: 100
  num_scales: 3
  dec_layers: 9
  nf: 512
  hidden_dim: 256
  refine_norm: spectral | batch | none
```

## 6. Data pipeline

### Grayscale pair generation

Critical: the model consumes a 3-channel RGB image whose RGB equals the LAB-L-derived grayscale (not standard luma). This must match between training and inference.

```
RGB uint8 → /255 float32 → cv2.cvtColor(RGB→LAB)
                                    │
                                    ├─► L_full (full-res, kept for re-merge at inference)
                                    │
                                    ▼ concat(L, 0, 0)  → "gray-LAB"
                                    ▼ cv2.cvtColor(LAB→RGB)
                                    │
                              RGB-of-gray  ← model input
                              AB           ← model target
```

### Recursive dataset

- `pathlib.Path.rglob` for `*.jpg|jpeg|png|webp|bmp|tif|tiff` (case-insensitive).
- Manifest cached to `<data_root>/.coliraz-manifest.txt`; rebuilt when root mtime changes.
- Random crop + horizontal flip during training; center crop for val/preview.
- Skip images smaller than `input_size` on either dimension (logged, not errored).
- cv2 backend, PIL fallback for esoteric formats.

### Val/preview holdout

`val_fraction` (default 0.01) of all paths reserved, picked deterministically by `hashlib.md5(path)` so the same images stay held-out across runs. Preview's 4 fixed samples are the first 4 by hash; random preview samples are re-drawn each tick from the same pool.

### Config

```yaml
data:
  root: ???                          # required
  val_fraction: 0.01
  num_fixed_preview_samples: 4
  num_random_preview_samples: 2
  augment:
    hflip: true
    rotate90: false
    color_jitter: false              # applied to RGB BEFORE grayscale derivation
  loader:
    batch_size: 32                   # or "auto"
    num_workers: 16
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4
```

## 7. Modular loss system

A registry + a `LossSet` composer. The trainer never branches on which losses are enabled.

### Contract

```python
@dataclass
class LossContext:
    pred_ab:   Tensor
    gt_ab:     Tensor
    pred_rgb:  Tensor
    gt_rgb:    Tensor
    gray_rgb:  Tensor
    discriminator: nn.Module | None = None

class ColorizationLoss(nn.Module):
    name: str
    def forward(self, ctx: LossContext) -> Tensor: ...
```

### Registry & available losses

| Registered name | Module | Notes |
|---|---|---|
| `l1_ab` | `losses/pixel.py` | L1 on AB |
| `l2_ab` | `losses/pixel.py` | MSE on AB |
| `charbonnier_ab` | `losses/pixel.py` | smooth L1 variant |
| `perceptual_vgg16bn` | `losses/perceptual.py` | VGG16-bn perceptual + optional style; lazy-loads VGG only when enabled |
| `gan` | `losses/gan.py` | Generator-side GAN loss; reads `ctx.discriminator`. Trainer pairs with a separate discriminator step. |
| `colorfulness` | `losses/colorfulness.py` | Encourages chroma |

### YAML configuration

```yaml
losses:
  - { name: l1_ab,             weight: 0.1 }
  - { name: perceptual_vgg16bn, weight: 5.0, config: { layer_weights: { conv1_1: 0.0625, conv2_1: 0.125, conv3_1: 0.25, conv4_1: 0.5, conv5_1: 1.0 }, criterion: l1 } }
  - { name: gan,               weight: 1.0, config: { gan_type: hinge, discriminator: { type: unet, nf: 64 } } }
  # commented loss = not loaded
  # - { name: colorfulness, weight: 0.5 }
```

### Presets

| Preset | Composition |
|---|---|
| `minimal` | `l1_ab @ 1.0` |
| `standard` | `l1_ab @ 0.1`, `perceptual_vgg16bn @ 5.0`, `colorfulness @ 0.5` |
| `ddcolor_full` | `l1_ab @ 0.1`, `perceptual_vgg16bn @ 5.0`, `gan @ 1.0`, `colorfulness @ 0.5` |
| `stable` | `charbonnier_ab @ 0.1`, `perceptual_vgg16bn @ 5.0` |

Used as `losses: !preset standard`.

## 8. Trainer

Single class, ~400 lines. No inheritance, no plugin system.

### Per-step

```python
def _train_step(self, batch):
    gray_rgb = batch["gray_rgb"].to(device, non_blocking=True, memory_format=channels_last)
    gt_ab    = batch["gt_ab"].to(device, non_blocking=True)
    L_full   = batch["L"].to(device, non_blocking=True)

    self.opt_g.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        pred_ab = self.model(gray_rgb)
        ctx = build_ctx(pred_ab, gt_ab, L_full, gray_rgb, self.disc)
        total_g, log_g = self.loss_set(ctx)

    if self.scaler:
        self.scaler.scale(total_g).backward()
        self.scaler.unscale_(self.opt_g)
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt_g); self.scaler.update()
    else:
        total_g.backward()
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt_g.step()

    if self.disc:
        self._discriminator_step(ctx)

    if self.ema: self.ema.update(self.model)
    self.scheduler_g.step()
    self._record_metrics(log_g, ...)
```

### Blackwell defaults

| Switch | Default | Rationale |
|---|---|---|
| `amp` | `bf16` | Blackwell bf16 matmul on par with fp16; fp32-range exponent → no GradScaler, no overflow debugging |
| `memory_format` | `channels_last` | ConvNeXt depthwise convs ~1.4× faster |
| `tf32` | on | `torch.backends.cuda.matmul.allow_tf32 = True` |
| `cudnn.benchmark` | on | static shapes → cuDNN picks fastest kernel |
| `optimizer` | fused AdamW | 5–10% faster |
| `compile` | off (opt-in) | `torch.compile(mode="default")`; user can set `max-autotune` |
| `gradient_checkpointing` | off | not needed at 96 GB unless large+512² with very large batch |
| `grad_accum_steps` | 1 | configurable |
| `clip_grad_norm` | 1.0 | standard stability |
| `ema_decay` | 0.999 | fp32 weight tracking |

### Training-time options surfaced

- Scheduler: `cosine` (warmup + cosine decay) | `multistep` (paper) | `constant`.
- `batch_size: auto` — fwd/bwd pass with growing batch until OOM, step back one.
- Resume from `runs/<run>/ckpt/last.pt`.
- Keys: `q` clean-shutdown, `s` checkpoint now, `p` preview now.

## 9. Live UI

### Rich dashboard

```
╭──────────────────────── coliraz train ─────────────────────────╮
│ run: tiny_run_2026-05-11_14-32   ckpt: runs/tiny_run/...      │
│ device: NVIDIA RTX 6000 Blackwell (95.7 GB)                    │
╰────────────────────────────────────────────────────────────────╯
╭─ progress ──────────────────────────────────────────────────────╮
│ step  12 480 / 400 000  ━━━━╺━━━━━━━━━━━━━━━━━━━━  3.1%  ETA 14h│
│ epoch 4 / ~128   it/s 18.3   img/s 586   lr 8.7e-5             │
╰────────────────────────────────────────────────────────────────╯
╭─ losses (EMA-30) ──────────────────╮ ╭─ gpu ───────────────────╮
│ total_g           0.4127  ▼ 0.0021 │ │ mem  43.2 / 95.7 GB     │
│ l1_ab             0.0312  ▼ 0.0001 │ │ util 97 %               │
│ perceptual_vgg    0.0631  ▼ 0.0008 │ │ temp 71 °C              │
│ gan_g             0.1840  ▲ 0.0034 │ │ pwr  582 / 600 W        │
│ d_total           0.6512  ▼ 0.0017 │ ╰─────────────────────────╯
╰────────────────────────────────────╯
╭─ last preview ──────────────────────────────────────────────────╮
│ wrote samples/latest.png @ step 12 400 (12s ago)                │
│ 4 fixed + 2 random   psnr 24.1   delta-e76 9.3                  │
╰────────────────────────────────────────────────────────────────╯
[ q quit  s save-now  p preview-now ]
```

- `rich.live.Live` at ~6 Hz.
- Losses EMA-smoothed over 30 steps; trend vs. 100 steps ago.
- GPU panel via `pynvml`; hides if unavailable.
- stdin keypress handler thread (non-blocking).
- All other stdout/stderr routed through `rich.console.Console` so they appear above the live region.

## 10. Sample preview

Background thread driven by a `threading.Event` set every `preview_every_s` (default 10 s). Renders off the train thread using a CPU snapshot of the model's `state_dict` — the EMA model when `ema_decay > 0`, otherwise the live training model. The render runs in `torch.inference_mode()` on a dedicated CUDA stream so the train step never blocks.

PNG layout (4 fixed + 2 random rows):

```
| original | grayscale | predicted | |Δ| ab heatmap |
```

- Caption strip: step, timestamp, EMA losses.
- Atomic write: render → `.latest.png.tmp` → `os.replace` → `latest.png`.
- Rotated history: every Nth tick (`preview_history_every`, default 10) also writes `samples/iter_NNNNNNN.png`.

## 11. Inference

`infer/pipeline.py` replicates the original `ColorizationPipeline` (same input/output contract) and adds:

- Batched folder inference (DataLoader over input dir).
- Aspect-preserving resize: AB output bilinearly upsampled to original H×W before merging with full-res L.

CLI:

```bash
coliraz infer --model ckpt.pt --input photo.jpg --output out.jpg
coliraz infer --model ckpt.pt --input ./bw --output ./out --input-size 512 --batch 8
```

## 12. ONNX export

- `torch.onnx.export` with opset 17 (lowest opset that supports native Attention op so ORT routes to a fused kernel).
- Dynamic axis: batch dim only; `input_size` baked in (re-export to change).
- Pipeline: export → `onnx.shape_inference.infer_shapes` → `onnxruntime.tools.symbolic_shape_infer` → `onnxsim.simplify` → `onnx.checker.check_model` → write.
- Automatic numerical parity: random input → PyTorch fp32 vs ORT fp32 → assert `max_abs_diff < 1e-3`. Fail loud on regression.
- Triggered automatically after training when `export.on_finish: true`, otherwise via `coliraz export`.

## 13. CLI & config

### Subcommands

```bash
coliraz train      --config configs/tiny.yaml --data /mnt/photos
coliraz train      --config configs/large.yaml --batch-size auto --compile
coliraz train      --config configs/tiny.yaml --resume runs/tiny_run/ckpt/last.pt
coliraz infer      --model ckpt.pt --input ./bw --output ./out
coliraz export     --model ckpt.pt --output ddcolor_tiny.onnx --input-size 512
coliraz scan-data  --root /mnt/photos
```

### Config layering

```
defaults  →  config-file YAML  →  !preset references  →  CLI flag overrides
```

CLI flags always win. All flags auto-documented by Typer.

### Configs

- `configs/default.yaml` — full surface with sensible defaults.
- `configs/tiny.yaml` — `model.size: tiny`, larger batch.
- `configs/large.yaml` — `model.size: large`, `losses: !preset ddcolor_full`, smaller batch.

## 14. Testing strategy

All tests run on CPU, full suite under 30 s.

| Test | What it verifies |
|---|---|
| `test_color.py` | `rgb→lab→rgb` round-trip within ε; LAB-L equals cv2 reference |
| `test_dataset.py` | Recursive scan finds all extensions, skips too-small, manifest cache works |
| `test_grayscale.py` | RGB-of-gray pipeline matches the original `ColorizationPipeline` byte-for-byte |
| `test_model_shapes.py` | DDColor forward shape and grad flow (tiny config, CPU, stubbed encoder where needed) |
| `test_losses.py` | Each loss returns a scalar with valid gradient on a fake context |
| `test_loss_set.py` | LossSet composer, preset expansion, has_gan detection |
| `test_train_step.py` | Single training step (tiny+CPU) completes and reduces loss on overfit batch |
| `test_checkpoint.py` | Save/load round-trip preserves model + optimizer state |
| `test_export_onnx.py` | Export → ORT load → fp32 parity with PyTorch within 1e-3 |
| `test_cli.py` | Typer commands parse with `--help`; `scan-data` runs end-to-end on tiny fixture |
| `test_preview.py` | Preview writer produces a valid PNG without blocking the test |
| `test_config.py` | YAML + `!preset` + CLI override merge semantics |

## 15. Dependencies (uv-managed)

`pyproject.toml` additions:

```toml
[project]
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
  "pynvml>=11.5",            # optional GPU stats
  "tqdm>=4.66",              # for scan-data progress only
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff>=0.6"]

[project.scripts]
coliraz = "coliraz.cli:app"
```

## 16. Migration steps (sequence overview)

The detailed implementation plan will be produced in the next step. High-level order:

1. Move `./DDColor/` → `./reference/ddcolor_original/`; update `pyproject.toml` deps; add scaffolding.
2. Implement utilities (color, gpu, timing) + tests.
3. Implement model modules (encoder → unet_blocks → pixel_decoder → color_decoder → refine → ddcolor → discriminator) + tests.
4. Implement losses (registry → each loss → LossSet) + tests.
5. Implement data pipeline + tests.
6. Implement trainer + UI + preview + checkpoint.
7. Implement inference pipeline + tests.
8. Implement ONNX export + parity test.
9. Implement CLI (Typer) + config layering + tests.
10. Smoke test: end-to-end `coliraz train` on a tiny fixture dataset for ~50 steps.

## 17. Open questions

None blocking implementation. Items deferred:

- Public DDColor checkpoint conversion (will write on request).
- FID/CF metrics (post-hoc against held-out folder if needed later).
- Multi-GPU (not needed for current target hardware).

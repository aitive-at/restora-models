# Temporal Old-Film Remaster — Design Spec

> **Status:** brainstorming converged 2026-05-17, awaiting user review.
> Supersedes `2026-05-16-latent-diffusion-refine-head-design.md` (whose
> per-frame premise is invalidated by the temporal-window decision below).
> The current B200 production checkpoint (per-frame model) is preserved
> only as a distillation teacher + warm-start init; it is not the
> deployment artifact going forward.

## 1. Goals and non-goals

### Goals

1. **Old-film remastering quality** competitive with DeepRemaster (2019),
   ColorMNet (2024), and BiSTNet on damaged b/w / faded archival footage,
   with one model that also handles modern restoration tasks
   (denoise / sharpen / dejpeg / deblur) on color footage.
2. **One model, configurable tasks** — the existing 5-axis task vector
   `(colorize, denoise, sharpen, dejpeg, deblur)` controls which
   restoration(s) the model performs per inference.
3. **Same contract across every model size** so they're interchangeable
   at the ONNX/C# layer. `forward(frames [B,7,3,H,W], config [B,5]) →
   rgb [B,3,H,W]`. Multiple sizes (nano / small / medium / large)
   produced via distillation from the large teacher; identical contract,
   different parameter counts.
4. **ONNX export at any input resolution** (dynamic spatial axes).
   TensorRT compatible (dynamic-shape engine with min/opt/max profiles).
5. **Single CLI** (`restora`) for train / infer / export / distill /
   bench / compare / gallery / data-prep, with old cruft removed.
6. **Keep the existing training harness** (trainer.py, preview.py,
   checkpoint.py, EMA, video-pair loader, RAFT flow precompute). They
   work, they're proven, and they're not the bottleneck.

### Non-goals

- **No reference color frames in v1.** Blind colorization only. The
  contract is designed to allow reference frames as a future v2
  add-on (extra optional input tensor) without breaking the v1 path.
- **No live-streaming compatibility.** The symmetric 7-frame window
  imposes 3 frames of look-ahead latency. Archival workflows can
  tolerate this; live ones cannot. Out of scope.
- **Not aiming for SOTA PSNR** on every axis. Diffusion-based hallucination
  trades fidelity for perceptual quality on hard axes (colorize,
  sharpen-8x). Easy axes (denoise, dejpeg, deblur) still hold up because
  the deterministic backbone produces a strong coarse output and the
  RSD refine head is gated by task and noise scale.
- **Not training our own VAE or text-conditioned diffusion**. RSD uses
  RGB-space residual prediction; no SD VAE, no CLIP/T5 deps.

## 2. Architecture overview

```
                    Input: 7 RGB frames (B,7,3,H,W)
                                  │
                                  ▼
                       ┌────────────────────┐
                       │ Distilled RAFT     │  ~5M params, frozen
                       │ (estimates flow    │  baked into the ONNX graph
                       │  from each of 6    │
                       │  neighbors → t)    │
                       └─────────┬──────────┘
                                  │ 6 backward flow fields
                                  ▼
                       ┌────────────────────┐
                       │ Flow-warp aligner  │  bilinear warp,
                       │ (warps 6 neighbors │  no params
                       │  into frame t      │
                       │  coordinates)      │
                       └─────────┬──────────┘
                                  │ 7 aligned frames concat → (B, 21, H, W)
                                  ▼
                       ┌────────────────────┐
                       │ TemporalNAFNet     │  config (B,5) injected
                       │ backbone           │  via FiLM at every block
                       │  - stem 21→nf      │
                       │  - 4 enc stages    │
                       │  - bottleneck:     │
                       │    NAFBlocks +     │
                       │    temporal-attn   │  attn only at 8× downsampled
                       │    (mem-safe at    │  bottleneck, any-resolution OK
                       │    any res)        │
                       │  - 4 dec stages    │
                       │  - Lab dual head   │  preserves dual-head Lab
                       │    (delta + ab-abs)│  trick
                       └─────────┬──────────┘
                                  │ coarse RGB (B,3,H,W) for frame t
                                  ▼
                       ┌────────────────────┐
                       │ RSD refine head    │  ~20M params,
                       │ (single-step       │  RGB-space residual diffusion
                       │  ResShift in RGB)  │  conditioned on config + t_inf
                       └─────────┬──────────┘
                                  ▼
                          Refined RGB (B,3,H,W)
```

Total inference params (large/teacher): **~110M** (5M RAFT + 55M
backbone + 20M refine head + heads/embed). Targets ~12–18 ms at 256² fp16
on B200, ~25–35 ms on RTX 6000 Blackwell. Smaller sizes via distillation:

| Size | Params | Backbone nf | Enc depths | Bottle | RSD width | Target speed (256² fp16 RTX 6000) |
|---|---|---|---|---|---|---|
| **nano** | ~8M | 24 | 1,1,1,2 | 2 | (no refine) | ~5 ms |
| **small** | ~22M | 36 | 2,2,2,4 | 4 | 64 | ~12 ms |
| **medium** | ~50M | 48 | 2,2,4,6 | 6 | 96 | ~22 ms |
| **large** | ~110M | 64 | 2,2,4,8 | 8 | 128 | ~35 ms |

## 3. Components in detail

### 3.1 Distilled RAFT

`src/restora_models/models/flow_distill.py` (new).

- **Architecture**: 4-stage encoder + 2-stage iterative refinement
  (ConvGRU-free static unroll for ONNX cleanliness). ~5M params at fp16
  = ~10 MB. Based on RAFT-Small but with the iterative loop fully
  unrolled to a fixed 4 iterations.
- **Training**: a one-shot, separate pre-training step. Teacher is our
  existing `_load_raft` (RAFT-Large, ~5.3M params but trained
  iteratively). Student is supervised with EPE on Sintel + a sample of
  our DAVIS / Vimeo data. ~10–20 hours on RTX 6000.
- **Why distill rather than use RAFT-Small directly**: RAFT-Small's
  iterative loop hurts ONNX cleanliness. Our static-unroll variant has
  exactly the same op surface as the rest of the model. Quality target:
  EPE within +0.5 of RAFT-Large on Sintel.
- **Frozen in the final ONNX**: weights baked in at export.
- **Fallback for v0.x**: until the distilled RAFT is trained, the
  backbone is trained with precomputed RAFT-Large flow (the path we
  already use). Export blocked until the distilled estimator is ready.
- **Test plan**: unit-tested for shape + EPE bound; trainer has a flag
  to use either precomputed flow (train) or in-graph flow (export).

### 3.2 Flow-warp aligner

`src/restora_models/models/warp.py` (new, ~50 lines).

- Pure bilinear grid_sample. No params. ONNX-supported via opset 16+.
- Input: 6 neighbor frames + 6 backward-flow fields tk→t.
- Output: 6 frames warped into frame-t coords. Concat with frame t to
  form a `(B, 21, H, W)` tensor.
- Out-of-bounds policy: zero-fill + a per-pixel visibility mask passed
  as an extra 7-channel tensor (one mask per frame). The backbone takes
  `(B, 21 + 7, H, W) = (B, 28, H, W)` as the actual stem input. The
  visibility mask lets the model down-weight occluded pixels.

### 3.3 TemporalNAFNet backbone

`src/restora_models/models/temporal_nafnet.py` (new).

- Stem: `Conv2d(28 → nf, 3×3)`. Replaces current `Conv2d(3 → nf)`. All
  rest of NAFNet is 2D-conv, fully resolution-independent.
- Encoder: 4 stages of `NAFBlock` (FiLM-conditioned, current
  `src/restora_models/models/nafblock.py`). Channel widths
  `nf, 2·nf, 4·nf, 8·nf`. Strided convs for downsampling (factor 2 per
  stage, total 16×).
- Bottleneck: 4–8 `NAFBlock`s (depending on size) PLUS a
  `TemporalSelfAttention` block: at the bottleneck the spatial is
  H/16 × W/16, so even for 1080p input that's 67×120 = 8k tokens —
  memory-safe for a single attention pass. Attention runs over the
  flattened spatial dim of the central-frame features only (temporal
  context comes from the warped concat at the stem).
- Decoder: mirror of encoder. Standard NAFNet skip connections.
- Output: Lab dual head (delta + ab-abs), same as current. Lab-delta
  output for all axes; ab-abs head zeros out for non-colorize axes via
  the existing colorize gate.

**Conditioning**: existing `ConfigEmbed` (`models/config_embed.py`,
5 → 128 MLP) produces `task_vec`, injected at every NAFBlock via FiLM
(unchanged).

### 3.4 RSD refine head

`src/restora_models/models/rsd_refine.py` (new, replaces
`models/heads.py:AdversarialRefineHead` and the deferred latent diffusion
head). Based on RSD (one-step ResShift, arxiv 2503.13358).

```python
class RSDRefineHead(nn.Module):
    """Single-step residual-shift diffusion in RGB space.

    No external VAE. Operates directly on the backbone's coarse RGB
    output, conditioned on the 5-axis task vector + a fixed inference
    timestep embedding. Lightweight (~20M params), fully convolutional,
    any-resolution, exports to a single ONNX subgraph.

    Train: predict the residual shift z → clean conditioned on noisy
    z_t = sqrt(alpha) · coarse + sqrt(1-alpha) · noise where alpha
    follows a small (~4 step) schedule sampled uniformly during training.

    Inference: single forward pass at a fixed t_inf chosen per-axis.
      - colorize / sharpen: t_inf = 0.3 (more hallucination)
      - denoise / dejpeg / deblur: t_inf = 0.05 (near-identity, the
        coarse is already strong)
    The task-conditioned UNet learns the per-axis behavior so a single
    fixed t_inf at inference works.
    """
```

Architecture: small UNet, 4 stages, residual blocks with FiLM
conditioning on `(config_vec, t_inf_emb)`. Width controlled by the size
preset (table in §2).

**Why RSD over latent-diffusion**: RSD operates in RGB space and is
fully convolutional, so it's resolution-agnostic without the SD VAE's
80M-param dependency. No `diffusers` dep. TensorRT-friendly. Single
forward pass, single ONNX subgraph.

### 3.5 Model contract

```python
forward(
    frames: torch.Tensor,   # (B, 7, 3, H, W), fp16/fp32, [0,1]
    config: torch.Tensor,   # (B, 5), [0,1] axis weights
) -> torch.Tensor:          # (B, 3, H, W), fp32, [0,1]
```

Constraints:
- `H, W` must be divisible by 16 (NAFNet's downsampling factor). The
  inference pipeline pads to a multiple of 16 internally and crops back.
- `B` is dynamic.
- The same forward works for any input resolution from 128² up to
  4K (memory-bounded by activation footprint, not by architecture).

## 4. Degradation pipeline (new)

`src/restora_models/data/degradations/` gets a new submodule for
old-film-specific degradations.

### 4.1 Preserved (current)

- `colorization.py` — RGB → grayscale via Lab L
- `denoise.py` — Gaussian + Poisson
- `superres.py` — bicubic 2x/4x/8x down/up
- `jpeg.py` — encode/decode Q ∈ [20, 70]
- `deblur.py` — Gaussian + motion blur

### 4.2 New (this design)

#### `film_overlay.py`
Composite real scratch / dust / grain textures from the DeepRemaster
asset pack (898 MB, 6,152 textures from `noise_data.zip`,
http://iizuka.cs.tsukuba.ac.jp/projects/remastering/data/noise_data.zip).

- Auto-download on first train via `restora download-film-overlays`
- Composite operation: `out = degraded + overlay_alpha · (overlay − 0.5)`
  with overlay rotation/scale/crop randomization per DeepRemaster Sec. 4.4
- Per-frame consistent across the 7-frame window for some texture types
  (grain, dust patterns persistent across frames) and per-frame
  randomized for others (transient scratches)
- Probability: applied with `p = 0.4` on colorize-axis batches, `p = 0.2`
  on other batches

#### `film_color_cast.py`
Sepia / cyan-fade / red-shift LUTs from a curated library + procedural
contrast crush + per-channel gamma drift. Applied with `p = 0.3` on
colorize-axis batches.

#### `gate_weave.py`
Per-frame sub-pixel translation jitter (the optical-printer "gate weave"
of physical film). U(-2, +2) px shift per frame, smoothed temporally
across the 7-frame clip. Applied with `p = 0.3` on video batches.

#### `mpeg_transcode.py`
ffmpeg subprocess: encode → decode at a random bitrate from
`mpeg1video` / `mpeg2video` / `h263`. Models VHS/broadcast-era
compression. Applied with `p = 0.2` on dejpeg-axis batches (and
optionally chained after JPEG for severe cases).

### 4.3 Application order (compound.py update)

```python
# Random subset of degradations sampled per-sample
# Order matters for realism:
clean → film_color_cast → film_overlay (grain/dust)
      → blur / down-up → noise → film_overlay (scratches)
      → jpeg / mpeg → output
```

The existing `data/compound.py` is extended; per-task axis_probs are
unchanged.

## 5. Training plan

### 5.1 Data

- **Image data** (unchanged): LAION / ImageNet / Open Images. Used for
  the "still" minibatch path (treated as a degenerate 7-frame clip via
  edge-replicate).
- **Video data (new): Vimeo Septuplet** at `~/data/vimeo-septuplet/`
  (~91K clips × 7 frames at 448×256, ~33 GB). Replaces DAVIS as the
  primary video source. Existing DAVIS path is kept as a smaller eval
  set and for backward-compat smoke tests.
- **Existing video data** (DAVIS, synthetic videos) stays as additional
  variety + eval.
- **Film overlay asset pack**: DeepRemaster `noise_data.zip` at
  `~/data/film-overlays/` (898 MB, auto-downloaded).

### 5.2 Curriculum

**Stage 0 — distilled RAFT** (~10–20 h on RTX 6000): one-shot
pre-training of the static-unroll RAFT student. Frozen thereafter.

**Stage 1 — backbone only, with precomputed flow** (~50 h on B200,
~120 h on RTX 6000): trains the TemporalNAFNet backbone +
dual head, no RSD refine. Uses RAFT-Large precomputed flow (the
existing path). Warm-start: load matching layer weights from the current
500k-step per-frame checkpoint where shapes align (encoder stages,
bottleneck NAFBlocks, decoder stages, dual head — only the stem and
temporal-attn block start fresh).

**Stage 2 — add RSD refine head** (~30 h on B200): freeze backbone,
train RSD only. Lighter activation footprint so batch can be larger.

**Stage 3 — end-to-end fine-tune** (~20 h on B200, optional): unfreeze
everything except the distilled RAFT. Lower LR. Resolves any
backbone↔refine fight.

**Stage 4 — distill to smaller sizes** (~30 h on RTX 6000 per size):
SLKD-style distillation (response L1 + intermediate feature MSE at
3 decoder stages + LPIPS perceptual + GAN-distill on hard axes).

### 5.3 Loss design

Existing `losses/registry.py` adds these new components:

| Component | Weight | What | Apply to |
|---|---|---|---|
| `temporal_pair` (existing) | 0.5 | Flow-warped consistency between adjacent frames within the 7-clip | All axes |
| `central_flicker` (new) | 0.3 | L1 between the model's output for clip[0:7] vs clip[1:8] at the overlapping center frame | All axes |
| `lpips_decoded` (new) | 0.4 | LPIPS perceptual on RGB output | All axes (replaces VGG perceptual) |
| `feat_match` (new, distill only) | 0.5 | MSE between teacher / student decoder features at 3 stages | Distill stage |

Everything else (chroma_lab, colorfulness, freq_l1, l1_pixel) is
preserved with the current weighting.

### 5.4 Optimizer

- **Backbone**: Muon (γ = 0.95) with lr=1e-3, weight_decay=0.01.
- **Norms + biases + RSD head**: AdamW (default).
- **Distilled RAFT**: AdamW, OneCycle.

Add `muon-pytorch` dep. ~1.5–2× faster convergence on restoration
backbones per the 2025 reports. Concrete local-training speedup is the
explicit deliverable here.

### 5.5 Validation pass criteria

After Stage 1:
- PSNR on a fixed Vimeo Septuplet eval split must hold ≥ current
  per-frame model's PSNR on the equivalent task within ±0.5 dB. The
  upper bound matters more — we want a meaningful improvement on
  damaged content.
- Visual A/B on a curated "old film" eval set (~50 clips, sourced
  from archive.org public-domain pre-1928 films): manual review of
  scratch fill, color plausibility, flicker stability.
- Inference latency on RTX 6000 at 256² fp16 must be < 50 ms.

## 6. Inference

```python
def forward_inference(self, frames_7, config):
    flows = self.flow_estimator(frames_7)               # 6 flows
    warped = self.warp(frames_7, flows)                 # (B, 7, 3, H, W) warped to t
    mask = self.visibility(flows)                        # (B, 7, H, W)
    stem_in = concat([warped.flatten(1,2), mask], dim=1) # (B, 28, H, W)
    coarse_rgb = self.backbone(stem_in, config)
    refined_rgb = self.rsd_refine(coarse_rgb, config, t_inf=self._t_for(config))
    return refined_rgb
```

Single ONNX file. Dynamic batch + spatial dims.

**Single-image path**: caller replicates the image 7× → identical
forward. The flow estimator outputs ~0 flow for identical frames,
which the warp passes through unchanged. Functionally degenerate to a
per-frame model on still inputs.

## 7. ONNX / TensorRT export

- ONNX opset 17 (current default).
- Dynamic axes: `{0: "batch", 3: "h", 4: "w"}` on input.
- Includes: distilled RAFT, warp (grid_sample), backbone, RSD head,
  Lab-RGB conversion. Single graph.
- `--task` baking unchanged: bakes the 5-axis config as a constant for
  consumers that only want one axis.
- TensorRT: dynamic-shape engine with min/opt/max profiles, e.g.
  `min=(1,7,3,256,256)`, `opt=(1,7,3,720,1280)`, `max=(1,7,3,2160,3840)`.
- fp16 default, fp32 supported, fp8 deferred until TRT 10.x has stable
  fp8-conv on Blackwell sm_120.
- Verification: `--verify-ep tensorrt` runs the exported engine end-to-end
  on a 7-frame replicate of a real image and asserts no CPU fallback.

## 8. CLI surface (final)

Single `restora` binary. The `restora-models` alias is dropped.

```
restora train       --config configs/{local,b200}-temporal.yaml ...
restora infer       --model X.pt --input clip-dir --output out-dir --color --denoise ...
restora export      --model X.pt --output X.onnx [--task colorize] [--precision fp16]
restora distill     --teacher X.pt --output student.pt --student-preset {nano,small,medium}
restora bench       --ckpt X.pt --input-size 256 --iters 100
restora compare     --ckpts X.pt Y.pt --data ~/data/vimeo-septuplet/val
restora gallery     --ckpt X.pt --data ~/data/vimeo-septuplet/val --axis colorize
restora prepare-data
   --vimeo ~/data/vimeo-septuplet/    # download + manifest Vimeo Septuplet
   --film-overlays ~/data/film-overlays/  # download DeepRemaster noise_data.zip
   --davis ~/data/davis/              # (existing, kept for eval)
   --imagenet/--openimages/...        # (existing, kept for still data)
   --precompute-flow                  # RAFT precompute on whatever video roots exist
```

Removed (vs current cli.py):
- `scan-data` — superseded by `prepare-data`
- `download` (LAION) — superseded by `prepare-data`
  (LAION still available via `--laion` flag inside prepare-data)
- `info` — folded into `bench --info` or just `restora export --dry-run`
- `download-davis`, `download-imagenet`, `download-openimages`,
  `prepare-videos`, `precompute-flow`, `make-synthetic-videos` —
  all subcommands of `prepare-data` instead, eliminating the top-level
  noise

The `restora-models` console-script alias is removed.

## 9. Compatibility with existing framework

- **Trainer (`train/trainer.py`)**: minor changes only. The model
  `forward(frames, config)` shape changes from `(B,3,H,W)` to
  `(B,7,3,H,W)`. The image loader wraps single images into 7-frame
  clips via edge-replicate (transparent to the trainer). Video loader
  already produces multi-frame clips; just becomes the primary loader.
- **Preview generation (`train/preview.py`)**: unchanged for the visual
  output; the model input adapter (7-frame wrap) is the only addition.
  Preview grids still render per-axis. The new model gets 4 rows added
  (colorize on a real Vimeo clip, denoise on a Vimeo clip, etc.) for
  temporal eyeballing.
- **Checkpoint (`train/checkpoint.py`)**: unchanged. EMA unchanged.
- **Configs**: new `configs/local-temporal.yaml` and
  `configs/b200-temporal.yaml`. Old `local.yaml` / `b200.yaml` removed
  (since per-frame model is replaced).
- **Tests**: new tests for `flow_distill`, `warp`, `temporal_nafnet`,
  `rsd_refine`. Existing tests for `nafblock`, `losses`, `degradations`
  unchanged.
- **Old model files**: `models/heads.py`, `models/diffusion_head.py`,
  `models/vae.py`, `models/discriminator.py` removed (the discriminator
  was already deprecated by the no-GAN diffusion plan). `models/nafnet.py`
  is gone — the new `models/temporal_nafnet.py` is the only backbone.

## 10. Cleanup pass

Removed in this PR (or follow-up):
- All NAFNet per-frame files (`nafnet.py`, `heads.py`, `vae.py`,
  `diffusion_head.py`, `discriminator.py`).
- Old configs (`local.yaml`, `b200.yaml`).
- `losses/diffusion.py`, `losses/gan.py` (no GAN, no SD-VAE diffusion).
- Deferred spec `2026-05-16-latent-diffusion-refine-head-design.md`
  marked superseded; left in place as history.
- Several plan docs that map to obsolete designs (kept as history under
  `docs/superpowers/plans/_archive/`).

Kept:
- All training-harness files (trainer, preview, checkpoint, ema, ui).
- All loss components except `gan.py` and `diffusion.py`.
- All data prep code, consolidated under `restora prepare-data`.
- All export code (extended for the new model's flow estimator).

## 11. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Distilled RAFT in the graph adds engine size + latency | Medium | Size is ~10 MB at fp16, latency ~3 ms at 256² — within target. Worst case, switch to teacher-RAFT precompute at training and ship two ONNX variants. |
| 7-frame edge-replicate on single images degrades quality vs a true per-frame model | Medium | The flow estimator returns ~0 for identical inputs and the warp is identity → degenerate per-frame path. Validate via PSNR on still images vs current model; tolerable if within −0.5 dB. |
| Vimeo Septuplet distribution doesn't match old film | Medium | Heavy degradation overlays + curated old-film eval set bridge this. Vimeo gives us temporal coherence training, not domain match. |
| Distillation gap to nano/small students > 2 dB on hard axes | High | SLKD-style feature matching + LPIPS is the explicit mitigation. Fall back: tighten the gap by allowing student to ship without the RSD head (just the deterministic backbone). |
| Muon optimizer fails to converge on this loss landscape | Low | Documented to work on restoration backbones. Fall back: AdamW. |
| TensorRT engine build is slow / fails on dynamic shapes | Medium | Fixed-shape engine ships as the default for production; dynamic-shape engine is an opt-in build. |
| Old-film evaluation is subjective | Medium | Defined eval set + LPIPS + manual A/B is the protocol. Document criteria explicitly. |

## 12. Open questions

These are explicit non-decisions deferred to implementation time:

- **Bottleneck temporal-attn vs. extra NAFBlocks**: empirically validate.
  Spec says include the attn block but if it gives <0.1 dB gain over
  more NAFBlocks at the same param count, replace with NAFBlocks.
- **Per-axis t_inf for RSD**: spec suggests 0.3 for hard / 0.05 for easy.
  Confirm empirically; consider learning per-axis t_inf as a small MLP
  output rather than a fixed table.
- **Visibility mask: 7-channel or learned**: spec uses a deterministic
  occlusion mask from flow consistency. Alternative is a learned
  attention. Start deterministic; revisit.
- **Stage-2 vs end-to-end from scratch**: Stage 1 + Stage 2 + Stage 3
  is the conservative plan. End-to-end from scratch might converge
  faster on the large model. Empirically decide after Stage 1.
- **Whether to ship a 7-frame-replicate single-image ONNX variant**:
  unnecessary if the degenerate path is good. Defer.

## 13. Build sequence (high level)

The implementation plan (separate doc) will sequence:

1. Distilled RAFT module + pre-training script + test.
2. Flow-warp + visibility-mask modules + tests.
3. TemporalNAFNet backbone + tests; load matching weights from the
   existing per-frame checkpoint (warm-start scaffolding).
4. RSD refine head + tests.
5. Film-overlay degradation module + auto-downloader.
6. Film-color-cast, gate-weave, mpeg-transcode degradation modules.
7. New losses (`central_flicker`, `lpips_decoded`, `feat_match`).
8. Vimeo Septuplet dataset loader + manifest format.
9. `configs/local-temporal.yaml`, `configs/b200-temporal.yaml`.
10. CLI consolidation (`prepare-data` umbrella, removals).
11. ONNX export wrapper update (includes distilled RAFT + warp).
12. Stage 0 (distilled RAFT), Stage 1 (backbone), Stage 2 (RSD),
    Stage 3 (end-to-end), Stage 4 (distillation).
13. C# integration smoke (7-frame buffering reference impl in the
    integration docs).
14. Cleanup pass: remove obsoleted files / configs / plans.

Each step is a separate PR / commit. The implementation plan will
spell out per-step file lists and test plans.

---

## Summary

Replace the per-frame model with a single temporal model:
**7-frame symmetric input** (edge-replicate for stills), **flow-warped
fusion with a distilled RAFT baked into the ONNX graph**, **TemporalNAFNet
backbone** (fully convolutional, any input resolution, FiLM-conditioned
on the existing 5-axis task vector), **RSD one-step refine head** in
RGB space (no SD VAE dep). One model contract across sizes
(nano/small/medium/large), one ONNX export, distillation via SLKD-style
feature matching + LPIPS. Old-film degradation pipeline ports
DeepRemaster's real-scratch/dust/grain overlay pack, adds color-cast +
gate-weave + MPEG transcode. Training uses Vimeo Septuplet primarily,
Muon optimizer on the backbone for local-Blackwell convergence speedup.
Existing trainer, preview, checkpoint, EMA, data prep all kept; CLI
consolidated to one `restora` binary, old per-frame artifacts removed.

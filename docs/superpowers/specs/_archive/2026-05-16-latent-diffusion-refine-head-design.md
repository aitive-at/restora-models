# Latent Diffusion Refine Head — Design Spec

> **Status:** design accepted 2026-05-16. Implementation deferred until
> the current production training run (commit `52c2c2f`, configs/b200.yaml)
> completes. The diffusion-refine model trains FROM the current run's
> final checkpoint, so we need that checkpoint as a starting point.

## 1. Goals and non-goals

### Goals

1. **Hallucinate believable detail** on the two hardest ill-posed tasks
   (colorize, sharpen-8x) where L1's regression-to-mean produces beige /
   blurry results.
2. **Drop-in replacement** for the current `AdversarialRefineHead`. Same
   public contract `(rgb, config) -> rgb`. Downstream C# integration
   doesn't change.
3. **Single forward pass at inference.** No iterative sampling, no
   scheduler. ONNX-exportable to one graph.
4. **Leverage a pretrained perceptual prior** without baking a 1B-param
   diffusion UNet into the deployment artifact. We use the SD 1.5 VAE
   only — its frozen latent space gives us perceptually-aware
   representations for free.

### Non-goals

- Not aiming for SOTA on PSNR — diffusion deliberately trades fidelity
  for perceptual quality on hard axes. Easy axes (denoise, dejpeg,
  deblur) still hold up because the deterministic backbone produces a
  strong coarse output that the diffusion head only refines.
- Not training a custom VAE. Stability AI's `sd-vae-ft-ema` is public,
  small enough (~80M), and proven on natural images.
- Not aiming for full multi-step diffusion. Single-step is a deliberate
  constraint for inference cost + ONNX simplicity.
- Not changing the backbone (NAFNet-large) or the deterministic dual
  output head. Those stay byte-identical.

## 2. Architecture overview

```
                                            (frozen, 40M)
                  RGB in (B,3,H,W)               │
                       │                         ▼
                       ▼               ┌──────────────────┐
                ┌──────────────┐       │ SD 1.5 VAE       │
                │ NAFNet-large │       │   encoder        │
                │   backbone   │       └────────┬─────────┘
                └──────┬───────┘                │
                       │                       latent
                       ▼                   (B,4,H/8,W/8)
                ┌──────────────┐                │
                │ Dual output  │ ──── coarse ───┤
                │   head       │       rgb      │   z_coarse
                │  (Lab-delta  │       │        │
                │   + ab-abs)  │       ▼        │
                └──────────────┘   VAE encode ──┤
                                   (same VAE)   │
                                                ▼
                                   ┌─────────────────────────┐
                                   │ LatentDiffusionRefine   │
                                   │ Head (single-step UNet) │
                                   │   - cond: config vec    │
                                   │   - cond: timestep emb  │
                                   │   - input: noisy latent │
                                   │   - input: z_coarse     │
                                   │   - input: features?    │
                                   │   ~25M params           │
                                   └────────────┬────────────┘
                                                │
                                          predicted latent
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │ SD 1.5 VAE      │
                                       │   decoder       │ (frozen, 40M)
                                       └────────┬────────┘
                                                │
                                                ▼
                                      Refined RGB (B,3,H,W)
```

Inference total: ~165M params, ~13ms at 256² fp16 on B200.

## 3. Components in detail

### 3.1 Frozen VAE wrapper

`src/restora_models/models/vae.py`:

```python
class FrozenSD15VAE(nn.Module):
    """Wrapper around the diffusers AutoencoderKL with frozen weights.

    Loaded once from stabilityai/sd-vae-ft-ema (cached via huggingface_hub).
    Scale factor 0.18215 is SD 1.5's canonical latent scaling; we apply
    it on encode and reverse on decode so the latent has unit-variance-ish
    statistics for the diffusion head.
    """
    SCALE = 0.18215

    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema",
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, rgb_01: torch.Tensor) -> torch.Tensor:
        # rgb in [0,1] -> [-1,1] (VAE input convention)
        x = rgb_01 * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.sample() * self.SCALE
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        rgb_m11 = self.vae.decode(z / self.SCALE).sample
        return (rgb_m11 + 1.0) / 2.0   # [-1,1] -> [0,1]
```

- **Dep:** add `diffusers>=0.30` to `pyproject.toml`. Pulls
  `transformers` transitively but only the AutoencoderKL class is used.
- **Memory:** ~80M fp32 params = 320 MB. With AMP bf16, ~160 MB on GPU.
  Negligible vs the 500 MB-3 GB activation footprint of the trainer.
- **Determinism:** we use `.sample()` from the encode's `latent_dist`
  (which adds Gaussian noise from the encoder's predicted mean+logvar).
  For training, this is the standard SD practice. For inference, we use
  `.mode()` (returns the mean, no sampling) for deterministic encoding.
- **ONNX export:** the VAE's encoder and decoder are independent
  modules. We export them as part of the full graph via the ONNX wrapper
  in `src/restora_models/export/wrapper.py` (extended).

### 3.2 LatentDiffusionRefineHead

Located in `src/restora_models/models/heads.py`, replacing
`AdversarialRefineHead`. Same public name and signature so the model
constructors don't need rewiring beyond changing the class.

Architecture (small conditional U-Net):

```python
class LatentDiffusionRefineHead(nn.Module):
    """Single-step diffusion head operating in SD 1.5 VAE latent space.

    Inputs to forward():
      backbone_features: (B, nf, H, W) — same nf as before (64). Spatially
                         downsampled to H/8 x W/8 via stride-8 conv before
                         concat into the latent UNet.
      coarse_rgb:        (B, 3, H, W) — deterministic dual-head output.
                         Encoded to z_coarse via the frozen VAE.
      config:            (B, 5) — task vector.

    Forward at training time:
      1. z_coarse = VAE.encode(coarse_rgb)
      2. z_clean  = VAE.encode(clean_rgb)              [target, training only]
      3. Sample t ~ U(0, 1)
      4. z_t = (1 - t) * z_coarse + t * eps,  eps ~ N(0, I)
      5. cond = AdaLN(MLP([config_emb, t_emb]))
      6. pred_z_clean = UNet(z_t, z_coarse, feat_proj, cond)
      7. loss = L1(pred_z_clean, z_clean) + lambda_pix * L1(VAE.decode(pred_z_clean), clean_rgb)

    Forward at inference:
      1. z_coarse = VAE.encode(coarse_rgb)
      2. eps = noise_source  (deterministic seed in production)
      3. z_t = (1 - t_inf) * z_coarse + t_inf * eps,  t_inf = 0.2
      4. pred_z_clean = UNet(z_t, z_coarse, feat_proj, cond)
      5. refined_rgb = VAE.decode(pred_z_clean)
      6. return refined_rgb
    """
```

UNet shape (~25M params, latent space is 8× cheaper than RGB space):

| Stage | Channels in | Channels out | Spatial (for 256² input) |
|---|---|---|---|
| stem (1×1 conv) | 4 (z_t) + 4 (z_coarse) + 16 (proj feat) = 24 | 96 | 32×32 |
| down 1 (2 AdaLN-resblocks + ↓2) | 96 | 192 | 16×16 |
| down 2 (2 AdaLN-resblocks + ↓2) | 192 | 384 | 8×8 |
| bottleneck (4 AdaLN-resblocks) | 384 | 384 | 8×8 |
| up 2 (2 AdaLN-resblocks + ↑2 + skip concat) | 384 + 192 | 192 | 16×16 |
| up 1 (2 AdaLN-resblocks + ↑2 + skip concat) | 192 + 96 | 96 | 32×32 |
| head (3×3 conv) | 96 | 4 (predicted latent) | 32×32 |

Each AdaLN-resblock:
```
adaLN(cond) -> SiLU -> conv 3×3 -> adaLN(cond) -> SiLU -> conv 3×3 -> +residual
```

Cond is the AdaLN-style projection of `(config_vec || timestep_emb)`:
- `config_vec`: (B, 5)
- `timestep_emb`: sinusoidal embedding of t, then 2-layer MLP, → (B, 128)
- Concat: (B, 133), then `Linear(133 → 384)` per block.

Backbone-feature projection: a single stride-8 conv that takes the
backbone's full-resolution feature tensor and downsamples to the
latent's 32×32 spatial.

```python
self.feat_proj = nn.Sequential(
    nn.Conv2d(nf, 32, kernel_size=8, stride=8),
    nn.SiLU(),
    nn.Conv2d(32, 16, kernel_size=1),
)
```

This gives the diffusion head pixel-aware context (without re-running
the backbone) so the model can decide when to be faithful (denoise) vs.
when to hallucinate (sharpen).

### 3.3 Why this architecture works for our use case

- **The VAE's latent space is the right unit of work for diffusion.**
  4×32×32 = 4096 cells per sample. RGB pixel space has 196608 (3×256²).
  The diffusion head sees a 48× smaller signal volume, which is both
  faster and easier to model.
- **Per-task gating happens through AdaLN.** The same network does
  different things for different config vectors — for colorize the
  AdaLN scales up the magnitude of high-frequency latent perturbations
  (hallucinating chroma detail); for denoise it's near-identity (the
  coarse already converged).
- **z_coarse + noisy z_t both as input** is essentially ControlNet's
  trick. The model has the deterministic anchor to fall back on, and
  the noise gives it freedom to deviate where useful.
- **Single fixed inference t.** Setting `t_inf = 0.2` gives the model
  80% of the way to the clean latent from the coarse, with 20% noise.
  Training over `t ∈ [0, 1]` teaches the model to handle this regime
  along with all others; single-step inference at the trained-on
  intermediate regime works because the model is multi-scale-trained.

## 4. Training plan

### 4.1 Curriculum

**Stage 0 (prereq):** the current 500k-step production run finishes.
Final checkpoint at `runs/<b200-run>/ckpt/final.pt` is the starting
point.

**Stage 1 (latent diffusion head only):** ~30-50h on B200.

- Freeze: NAFNet backbone, dual output head, VAE.
- Trainable: LatentDiffusionRefineHead only (~25M params).
- Total steps: ~200k.
- Batch size: 96 (much smaller activation footprint since the diffusion
  head operates in the small latent space).
- Loss stack (see §4.2).
- Optimizer: AdamW, lr=2e-4 with 5k step warmup.

**Stage 2 (end-to-end fine-tune):** ~20h on B200, optional.

- Unfreeze: backbone + dual head + diffusion head.
- VAE stays frozen always.
- Lower LR: 5e-5 for backbone, 2e-4 for diffusion head.
- 50k additional steps.
- Loss weights re-tuned to preserve deterministic-axis fidelity.

Stage 2 is optional — Stage 1 alone should already give the
hallucination win on colorize/sharpen. Run Stage 1, validate, decide
whether to do Stage 2 based on whether the easy axes regressed.

### 4.2 Loss design

Total loss = sum of:

| Component | Weight | What it does | Apply to |
|---|---|---|---|
| `l1_latent` | 1.0 | L1 between predicted z_clean and target z_clean | All axes |
| `l1_rgb_decoded` | 0.5 | L1 between VAE-decoded prediction and ground-truth RGB | All axes |
| `perceptual_vgg16bn` | 0.5 | VGG perceptual loss on decoded RGB | All axes |
| `chroma_lab` | 0.20 | Lab chroma loss on decoded RGB | colorize only |
| `colorfulness` | 0.10 | Colorfulness loss on decoded RGB | colorize only |
| `freq_l1` | 0.40 | Frequency-domain L1 on decoded RGB | sharpen only |
| `temporal_pair` | 0.5 | Existing flow-warped consistency loss | Video pairs only |

Notes:
- `l1_latent` is the core diffusion training signal. Direct supervision
  on the predicted latent.
- `l1_rgb_decoded` ensures the VAE-decoded output stays close to ground
  truth in pixel space. Without this, the latent prediction could be
  "correct in latent space but produce visual artifacts after decode".
- The chroma/colorfulness/freq weights are bumped *up* compared to the
  current recipe (was 0.15 / 0.05-0.08 / 0.30) because the diffusion
  head's job is precisely to make these axes vivid. Higher gradient
  pressure here exploits the hallucinatory capacity.
- **No GAN loss.** The diffusion training signal subsumes adversarial
  training. Tested in the LCM/SD-Turbo papers; we follow that lead.
- **`identity_prob = 0.05`** stays. The model must learn to leave
  identity-config inputs alone.

### 4.3 Timestep sampling

`t ~ Uniform(0, 1)` during training. We do NOT bias toward small t —
the model needs to learn the full noise→clean map so single-step at
intermediate t works.

Optional refinement: importance-sample t to spend more compute near
the inference operating point. `Beta(1.5, 1.5)` peaks at 0.5 and gives
more density in `[0.1, 0.9]`. Validate empirically; default uniform.

### 4.4 Data and degradations

Identical to current production training. Same axis_probs (0.65
colorize, 0.50 others). Same degradation parameter ranges. The
VAE latent has different statistics from RGB but the degradations are
applied in RGB space upstream.

### 4.5 Validation and pass criteria

After Stage 1, compare against the current production model on:

1. **Per-axis PSNR delta** via `restora compare`.
   - colorize: must improve or hold ≥ -0.5 dB (perceptual axes can lose
     PSNR without losing quality)
   - sharpen: must improve
   - denoise / dejpeg / deblur: must hold ≥ -0.3 dB (these are mostly
     deterministic — small loss is OK from latent roundtrip).
2. **Visual A/B comparison** on the preview grids. Manual inspection of
   colorize-only and sharpen-8x outputs side-by-side.
3. **Temporal stability** on DAVIS clips at inference. The model must
   not flicker visibly when fed consecutive frames.

If criteria fail: bisect to find which loss component is mis-weighted.
Most likely failure mode: latent decoded outputs are slightly soft for
easy axes — fix by bumping `l1_rgb_decoded` weight.

## 5. Inference

```python
@torch.inference_mode()
def forward_inference(self, rgb_in, config):
    coarse_rgb = super().forward(rgb_in, config)        # backbone + dual head
    z_coarse = self.vae.encode_mode(coarse_rgb)         # deterministic encode
    z_t = (1.0 - self.t_inf) * z_coarse + self.t_inf * self._noise_for(z_coarse)
    pred = self.refine_unet(z_t, z_coarse, self.feat_proj(features), config, self.t_inf)
    refined_rgb = self.vae.decode(pred)
    return refined_rgb.clamp(0.0, 1.0)
```

### Determinism vs hallucination knob

The `_noise_for(z_coarse)` function controls the noise source:

| Mode | What it does | Use case |
|---|---|---|
| `random` | `torch.randn_like(z_coarse)` | Per-image standalone restoration (e.g. one-off photo) |
| `seeded(frame_idx)` | `torch.Generator().manual_seed(idx)` | Reproducible across runs, varies per frame |
| `fixed_pattern` | Same noise tensor across frames | Best video temporal stability — adjacent frames see same noise, denoise consistently |
| `zero` | `torch.zeros_like` | Degenerate: model just decodes z_coarse through VAE. Roughly equivalent to "no diffusion". |

Default for video: `fixed_pattern`. The C# downstream gets a knob to
choose.

### t_inf choice

`t_inf = 0.2` is the recommended default. Lower (e.g. 0.05) → less
hallucination, closer to deterministic. Higher (e.g. 0.5) → more
hallucination, more variation, potentially less faithful. Make it a
runtime hyperparameter (extra ONNX input or fixed-at-export).

## 6. ONNX export

The current export pipeline (`src/restora_models/export/onnx.py`) needs
extending:

1. Bake `t_inf` and noise source mode at export time, OR add them as
   extra ONNX inputs for runtime control. **Recommendation:** bake
   `t_inf=0.2` and noise mode `fixed_pattern`; consumers who need
   variation re-export with different bakes. Keeps the C# consumer's
   contract identical (2 inputs in, 1 out).
2. The VAE module is included in the graph. ONNX export of
   `AutoencoderKL` is a known-supported path in `diffusers`.
3. fp16 / fp32 export modes both work; fp8 may not, since the VAE was
   trained at fp32 and its activations may overflow fp8. Default
   recommendation for production: fp16.

Resulting ONNX size: ~165M params at fp16 = ~330 MB. ~4× the current
exported size but still well within consumer GPU memory.

## 7. Compatibility with existing framework

- **Trainer:** no changes. `LatentDiffusionRefineHead` replaces
  `AdversarialRefineHead` at the same call site in `nafnet.py`. The
  trainer's `LossSet` adds the new loss components via existing
  registry pattern.
- **Configs:** new `configs/b200-diffusion.yaml` for Stage 1
  (Stage 2 may use a separate config). Existing `configs/b200.yaml`
  unchanged — current production run is the prereq.
- **CLI:** no new commands. `restora train --config b200-diffusion.yaml`
  works as-is.
- **Tests:** new `tests/test_latent_diffusion_head.py` for unit
  coverage of the new head. Existing `tests/test_adversarial_refine_head.py`
  is kept temporarily for regression — delete after the new architecture
  is validated.
- **ONNX consumer (C# guide):** unchanged. Same 2-input / 1-output
  contract. The C# code in `docs/integration/csharp-video-inference.md`
  works against the new ONNX without modification.

## 8. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| VAE roundtrip is too lossy for easy axes (denoise PSNR drops > 1 dB) | High | Bump `l1_rgb_decoded` weight, or add a per-axis gate that skips the diffusion path for non-(colorize, sharpen) configs at inference time |
| Diffusion head doesn't converge to single-step quality | Medium | Add 4-step inference path (still fast, allowed by the same model) as a fallback. Or use Consistency Distillation in a follow-up |
| Pretrained VAE's training distribution doesn't match degraded images | Low | VAE was trained on LAION-2B; degraded inputs are sub-distribution but VAE encoder handles them gracefully in practice |
| ONNX export of VAE fails or is slow at runtime | Medium | Export VAE encode/decode separately; consumer chains them with the DRH between. Standard diffusers pattern, well-tested |
| Adding diffusers dep bloats install | Low | diffusers + transformers add ~150 MB of deps. Acceptable for a research codebase; doesn't affect inference runtime |
| Stage 2 (end-to-end FT) regresses easy axes | Medium | Skip Stage 2 if Stage 1 result is good. Stage 1 alone is the conservative win |

## 9. Open questions

These are explicit non-decisions deferred to implementation time:

- **Should we use `latent_dist.sample()` or `.mode()` during training?**
  Sample (stochastic) is standard SD practice; mode (deterministic)
  reduces noise in the training signal. Pick empirically — try mode
  first; if convergence is slow, switch to sample.
- **Inference t schedule.** Fixed t=0.2, or learned per-axis?
  Conditional t (e.g. higher for colorize than for deblur) might give
  better task-specific hallucination. Default fixed; revisit.
- **Should `t_inf` be a runtime ONNX input?** Adds flexibility for the
  C# consumer to tune at deploy time. Default no (bake it), revisit
  based on consumer feedback.
- **VAE choice.** `sd-vae-ft-ema` is the conservative pick. Newer SDXL
  VAE (`madebyollin/sdxl-vae-fp16-fix`) has slightly better fidelity.
  Switching is a 1-line change; A/B during validation.
- **Composability with the current adversarial refine head.** Spec says
  "replace entirely" per user direction, but the dual-output head still
  produces a coarse RGB which the diffusion head refines. The current
  adversarial refine head's *interface* is replaced; the deterministic
  dual-head stays.

## 10. Build sequence (high-level)

When ready to implement (post-production-run-finish), the order is:

1. Add `diffusers` dep, write `FrozenSD15VAE` wrapper + unit tests.
2. Write `LatentDiffusionRefineHead` class + unit tests for forward,
   backward, AdaLN conditioning.
3. Add loss components (`l1_latent`, `l1_rgb_decoded`) to the loss
   registry.
4. Write `configs/b200-diffusion.yaml` for Stage 1 training.
5. Modify `NAFNetMultiTask.__init__` to instantiate the new head when
   `cfg.model.refine_type == "diffusion"`.
6. Run Stage 1 training, validate, decide on Stage 2.
7. Extend ONNX export wrapper to include VAE in the graph.
8. Run regression on C# downstream against the new ONNX.

Each step is a separate PR / commit. A dedicated implementation plan
(`docs/superpowers/plans/2026-05-XX-latent-diffusion-refine-head.md`)
should be written before code lands; that's where the per-step file
list and test plan go.

---

## Summary

NAFNet-large + dual output head stay untouched. Replace the
`AdversarialRefineHead` with a `LatentDiffusionRefineHead` that:

1. Encodes the deterministic coarse output into SD 1.5 VAE's latent
   space (frozen pretrained VAE).
2. Adds noise per a single-step linear-interpolation schedule.
3. Predicts the clean latent in one UNet forward pass, conditioned on
   the 5-axis task vector + timestep embedding.
4. Decodes the prediction back to RGB through the same VAE.

Trains from the current 500k-step deterministic checkpoint. ~30-50h of
additional B200 compute. ~165M params in the final ONNX (still
exportable + C#-loadable). Single forward pass at inference, ~13ms at
256² fp16.

Gains: hallucinated detail on the hard ill-posed axes, leveraging the
SD VAE's perceptual prior. Risks: VAE roundtrip cost on easy axes;
mitigated by reconstruction loss weighting.

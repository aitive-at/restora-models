# Verifying the model is actually improving

Before committing to a 60-hour production run, you want to know that the
training loop is moving PSNR in the right direction. The repo has all
the pieces — this doc just stitches them together into a recipe.

## The recipe in 30 seconds

```sh
# 1. Train a smoke run with checkpoint history enabled
uv run restora train --config configs/local.yaml

# 2. Compare an early checkpoint against the final one
uv run python scripts/eval_checkpoints.py \
  --ckpts runs/<smoke>/ckpt/iter_0001000.pt runs/<smoke>/ckpt/final.pt \
  --data ~/data/laion-images \
  --n 64 --seed 0
```

**Pass criteria: all 5 axes show positive Δ PSNR.** Anything else = stop
and debug before scaling up.

## What `eval_checkpoints.py` actually measures

The script (scripts/eval_checkpoints.py) builds a deterministic eval set:
- N images sampled with `seed=0` from `--data` root.
- Each image is degraded under each of the 5 single-axis configs with
  fixed parameters (σ=0.03 for denoise, factor=4 for sharpen, Q=40 for
  dejpeg, blur σ=2.0 for deblur, identity for colorize) — so noise is
  removed from the comparison.
- For each checkpoint, runs forward once per axis and computes mean PSNR
  against the clean target.
- Reports per-axis Δ between the second and first checkpoint, with arrows
  (▲ improved, ▼ regressed, · unchanged).

Why fixed parameters and not random: removes noise between the two
checkpoint evaluations so a small real improvement isn't swamped by
sample-to-sample variance.

## Expected output shape

```
[eval] building eval batch from ~/data/laion-images: n=64 size=256

[eval] iter_0001000.pt  step=1000
   colorize:  21.13 dB
    denoise:  29.45 dB
    sharpen:  24.10 dB
     dejpeg:  27.62 dB
     deblur:  25.88 dB

[eval] final.pt  step=5000
   colorize:  23.78 dB
    denoise:  29.82 dB
    sharpen:  26.45 dB
     dejpeg:  28.91 dB
     deblur:  27.10 dB

[eval] delta vs iter_0001000.pt:
  final.pt:
       colorize: +2.650 dB ▲
        denoise: +0.370 dB ▲
        sharpen: +2.350 dB ▲
         dejpeg: +1.290 dB ▲
         deblur: +1.220 dB ▲
```

## What to do when it doesn't look like that

### One axis is flat or negative
A specific axis getting no signal usually means:
- **colorize flat**: `chroma_lab.weight` too low, or `axis_probs.colorize`
  too low. Bump chroma_lab from 0.15 → 0.20-0.25 and re-smoke.
- **sharpen flat**: `freq_l1.weight` too low. The configs ship 0.30 — proven on
  the iter-6 results. If still flat, check that `sharpen_factor` actually
  varies (factor_choices: [2, 4, 8] in the config).
- **denoise regressed**: probably means perceptual loss is overpowering
  L1. Lower `perceptual_vgg16bn.weight` from 0.5 → 0.3.

### All axes are flat
Worse signal — the training loop isn't learning anything:
- Check the run log for `_skipped_grad` rate. >5% = lr too high or
  gradient instability.
- Check loss curves: if `total_g` is flat from step 0, the model may not
  be receiving gradient (e.g. forgot to enable AMP, or fp32 with very
  small lr).
- Check `grad_norm` in the log: should be in the [0.1, 10] range. If
  always zero, gradients aren't flowing.

### All axes regressed
The model unlearned. Common causes:
- GAN warmup started before reconstruction losses converged. Look at
  `train.gan_warmup_start` — should be at least 20% into training.
- `temporal_pair` weight too high relative to image losses; video
  consistency was pulling the model away from per-image fidelity.

## Comparing later in production

The B200 config writes `iter_NNNNNNN.pt` every 50k steps. Mid-run sanity
check after ~200k steps:

```sh
uv run python scripts/eval_checkpoints.py \
  --ckpts runs/<run>/ckpt/iter_0050000.pt runs/<run>/ckpt/iter_0200000.pt \
  --data /workspace/data/images/relaion2B-multi-aesthetic \
  --n 64 --seed 0
```

This is your "is the long run still climbing?" check. If 200k > 50k on
all 5 axes, let it ride. If not, snapshot the run state, decide whether
to keep training or revisit the recipe.

## A note on PSNR vs perceptual quality

PSNR is a fidelity metric, not a quality metric. The model may be making
images *visually* better while PSNR only moves a fraction of a dB —
especially on colorize (where there are many "correct" colorings) and
sharpen-8x (where hallucinating high-frequency detail trades PSNR for
perceived sharpness). For those axes, also eyeball the `samples/iter_*.png`
preview grids — they're the qualitative check `eval_checkpoints.py` can't
do automatically.

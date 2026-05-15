# restora-models

Multi-task image restoration: one model trained jointly on **colorization**,
**super-resolution**, **denoising**, **deblurring**, and **JPEG-artifact
removal**. A single 5-axis conditioning vector tells the model which
restoration to perform per image (or per axis combination per image).

**Architecture:** NAFNet-large backbone (33M params) with Lab-native dual
output head (Lab-delta + ab-abs, gated by the colorize axis) and an
adversarial refinement head (6.5M params) trained with GAN + perceptual
losses for the hard ill-posed tasks (colorize, sharpen-8x). Temporal
consistency is enforced during training via DAVIS-2017 video-pair batches
with a flow-warped `temporal_pair` loss; inference is per-frame.

**Inputs/outputs:**
- Input: RGB tensor `(B, 3, H, W)` in [0, 1] + 5-axis config vector
  `(B, 5)` `[colorize, denoise, sharpen, dejpeg, deblur]`.
- Output: RGB tensor `(B, 3, H, W)` in [0, 1].

ONNX export is per-task (one file per task with the config baked in, e.g.
`colorize.onnx`, `sharpen.onnx`) or generic (one file, two inputs). fp32
/ fp16 / fp8 precision modes are first-class.

## Quick start

```sh
# 1. Install (single binary uv: https://astral.sh/uv)
uv sync

# 2. Download a training image dataset. Three options:
#    - LAION-aesthetic (URL-based, may stall on flaky CDNs):
uv run restora download --output ~/data/laion
#    - ImageNet-1k via HF (gated, bundled parquet, fast CDN):
uv run python scripts/download_imagenet1k.py --out ~/data/imagenet
#    - Open Images via AWS S3 mirror (no auth, fast):
uv run python scripts/download_openimages.py --out ~/data/openimages --split validation

# 3. Prepare video data for temporal training (DAVIS + RAFT flow):
uv run python scripts/prepare_video_dataset.py --out ~/data/laion-videos

# 4. Train (local smoke or B200 production):
uv run restora train --config configs/local.yaml          # ~45 min smoke
uv run restora train --config configs/b200.yaml --compile # 60h production

# 5. Verify model improved before scaling to production:
uv run python scripts/eval_checkpoints.py \
  --ckpts runs/<smoke>/ckpt/iter_0001000.pt runs/<smoke>/ckpt/final.pt \
  --data ~/data/laion-images --n 64 --seed 0

# 6. Export per-task ONNX after the long run:
uv run restora export --model runs/<run>/ckpt/final.pt --output colorize.onnx --task colorize
```

The CLI also responds to `restora-models` (the full project name) as an
alias for `restora`.

## Configs

Only two configs ship — one local smoke, one B200 production:

| Config | Hardware | Steps | Wall time | Purpose |
|---|---|---|---|---|
| `configs/local.yaml` | RTX PRO 6000 Blackwell | 5000 | ~45 min | Pipeline validation |
| `configs/b200.yaml` | B200 | 500000 | ~60 h | Production |

Same architecture, loss recipe, and axis probabilities; only batch size /
workers / schedule differ. That makes the local smoke result predictive
of how production will behave.

## Docs

- **`docs/integration/b200-deployment.md`** — end-to-end B200 deployment recipe.
- **`docs/integration/verifying-model-improvement.md`** — how to use
  `eval_checkpoints.py` to confirm the model is improving before the
  long run.
- **`docs/integration/laion-download.md`** — LAION-specific download notes.
- **`docs/integration/onnx-inference-guide.md`** — consumer-side inference.

## Design history

Design specs and plans live under `docs/superpowers/specs/` and
`docs/superpowers/plans/`. The iteration journey (training-quality
breakthrough, dual-head architecture, adversarial refine head, temporal
video pair training, per-task ONNX export) is captured in the commit log.

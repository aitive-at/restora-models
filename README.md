# restora-models

Multi-task image restoration: one model trained jointly on **colorization**,
**super-resolution**, **denoising**, **deblurring**, and **JPEG-artifact
removal**. A single 5-axis conditioning vector tells the model which
restoration to perform per image (or per axis combination per image).

Two interchangeable backbones, same `(rgb, config) → rgb` contract:

- **NAFNet** — fast, Lab-native, lightweight (6M tiny / 33M large)
- **PromptIR** — config-driven prompt-bank variant of the NeurIPS-2023
  PromptIR architecture (4M tiny / 26M large)

ONNX export is per-task (one file per task with the config baked in, e.g.
`colorize.onnx`, `sharpen.onnx`) or generic (one file, two inputs). fp32
/ fp16 / fp8 precision modes are first-class.

## Quick start

```sh
# 1. Install (single binary uv: https://astral.sh/uv)
uv sync

# 2. Download LAION-aesthetic images (HF auth required)
uv run restora download --output ~/data/laion

# 3. Train
uv run restora train --config configs/laion-compound.yaml \
                     --data ~/data/laion/images/relaion2B-multi-aesthetic

# 4. Export per-task ONNX
uv run restora export --model runs/<run>/ckpt/final.pt --output colorize.onnx --task colorize
```

The CLI also responds to `restora-models` (the full project name) as an
alias for `restora`.

## Deploying on a fresh GPU box

See **`docs/integration/b200-deployment-checklist.md`** for the
`git clone → uv sync → train` recipe, plus a gotcha table for B200,
torch.compile warmup, fp8 capability detection, etc.

For LAION data specifically: **`docs/integration/laion-download.md`**.

## Design + history

This codebase replaces the colorization-only `coliraz` project. Design
specs live under `docs/superpowers/specs/`, including the 2026-05-13
compound-conditioning design (5-axis vector) and the 2026-05-14
dual-output head spec (Lab `ab` head gated by the colorize axis).

Iteration journey (training-quality breakthrough, dual-head architecture,
per-task ONNX export) is captured in the commit log.

# B200 SXM deployment guide

Target hardware (verified 2026-05-15):
- NVIDIA B200 (Blackwell, sm_100, ~183 GB HBM3e, 1000 W TDP)
- Driver `580.126.20` (CUDA 13.0 capable)
- Persistent data mounts: `/workspace/data` (images) + `/workspace/data-videos`
- Code checkout: `/workspace/code/restora-models`

The B200 is sm_100 (Blackwell SXM). Your current `torch 2.11.0+cu128` wheel
supports it via PTX JIT — the first forward pass on a new shape takes
~30s to compile kernels; subsequent passes are cached. **Do not upgrade
to a cu130 wheel before the production run** — see "CUDA upgrade" section
below.

End-to-end from cold checkout to live training in **6 commands**.

## 1. Bootstrap

```sh
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone + sync
cd /workspace/code
git clone https://github.com/<your-org>/restora-models.git
cd restora-models
uv sync

# Sanity check: GPU + torch happy
uv run python -c "import torch; \
  print('cuda:', torch.cuda.is_available()); \
  print('device:', torch.cuda.get_device_name(0)); \
  print('torch.cuda runtime:', torch.version.cuda); \
  print('compute cap:', torch.cuda.get_device_capability(0))"
# Expected:
#   cuda: True
#   device: NVIDIA B200
#   torch.cuda runtime: 12.8
#   compute cap: (10, 0)
```

## 2. HuggingFace auth

```sh
uv run huggingface-cli login   # paste your hf_... token
```

Then request dataset access (one-time, usually instant):
- <https://huggingface.co/datasets/laion/relaion2B-multi-aesthetic> (17M images)

## 3. Download LAION images → `/workspace/data`

The send-keys / multi-line tmux paste is fragile. Use Option 1 (type-it-yourself):

```sh
tmux new -s dl-laion
# inside tmux:
cd /workspace/code/restora-models
uv run restora download --output /workspace/data --dataset relaion2B-multi-aesthetic --processes 32 --threads 128 --timeout-s 5 2>&1 | tee /workspace/data/download.log
# Ctrl-B then D to detach
```

Reattach: `tmux attach -t dl-laion`. Disk: ~430 GB at 384px JPEG; ~8 h on 10 Gbps.

Resume: just re-run with the same flags — img2dataset skips completed shards.

## 4. Prepare video data → `/workspace/data-videos`

```sh
tmux new -s prep-video
# inside tmux:
cd /workspace/code/restora-models
uv run python scripts/prepare_video_dataset.py --out /workspace/data-videos 2>&1 | tee /workspace/data-videos.prep.log
# Ctrl-B then D to detach
```

This script does two things end-to-end:
1. Downloads DAVIS-2017 (~480 MB zip → ~9 GB extracted) and lays out frames.
2. Precomputes RAFT optical flow at training resolution (~30-50k pairs, GPU-bound on B200, ~30 min).

Re-runs skip both stages if outputs are present.

## 5. Build the image manifest (one-time, fast)

```sh
uv run restora scan-data --root /workspace/data/images/relaion2B-multi-aesthetic
```

Writes a manifest so training startup doesn't walk the tree from scratch.

## 6. Smoke test on B200 BEFORE the 60h run (recommended)

Before committing the production schedule, verify the pipeline is improving
the model. Run a 5k-step subset of production with the SAME architecture
and loss recipe, then compare early-vs-late checkpoints:

```sh
tmux new -s smoke
cd /workspace/code/restora-models
uv run restora train --config configs/local.yaml --data /workspace/data/images/relaion2B-multi-aesthetic --video-root /workspace/data-videos
# Ctrl-B then D to detach. ~10-20 min on B200 (vs 30-60 min on local).
```

After it finishes:

```sh
# Find the run directory
ls -1t runs | head -3

# Compare early vs final
uv run python scripts/eval_checkpoints.py \
  --ckpts runs/<smoke-run>/ckpt/iter_0001000.pt runs/<smoke-run>/ckpt/final.pt \
  --data /workspace/data/images/relaion2B-multi-aesthetic \
  --n 64 --seed 0
```

**Pass criteria: all 5 axes show positive Δ PSNR** (colorize, denoise,
sharpen, dejpeg, deblur). If any axis is flat or negative, do not start
the 60h run — debug the loss recipe first.

## 7. Production training (60h, 500k steps)

```sh
tmux new -s train
cd /workspace/code/restora-models
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run restora train --config configs/b200.yaml --compile
# Ctrl-B then D to detach
```

This:
- Loads `configs/b200.yaml` (bs=96, 500k steps, full pipeline).
- Enables `torch.compile` (Inductor on sm_100 is stable; first step ~3-5 min).
- Reads images from `/workspace/data/images/relaion2B-multi-aesthetic` and
  video pairs from `/workspace/data-videos`.

Live progress: `tmux attach -t train`. Rich UI shows per-task PSNR, loss
EMAs, GPU temp/util, throughput.

### Numbered checkpoints

The B200 config saves `iter_NNNNNNN.pt` every 50k steps (10 total). Use
these for late-run A/B comparisons — they don't get overwritten:

```sh
# After ~200k steps, sanity-check that the model kept improving:
uv run python scripts/eval_checkpoints.py \
  --ckpts runs/<run>/ckpt/iter_0050000.pt runs/<run>/ckpt/iter_0200000.pt \
  --data /workspace/data/images/relaion2B-multi-aesthetic \
  --n 64 --seed 0
```

## 8. Export ONNX after training

```sh
# Per-task ONNX (RGB in, RGB out) — what most consumers want
for task in colorize denoise sharpen dejpeg deblur; do
  uv run restora export \
    --model runs/<run>/ckpt/final.pt \
    --output ${task}.onnx \
    --task $task \
    --precision fp16
done
```

## CUDA upgrade — do NOT upgrade before production

Your driver supports CUDA 13.0 at the kernel level, but **PyTorch's runtime
is set by which wheel you install**, not by your driver. Current state:

| Layer            | Version       | Notes                                    |
|------------------|---------------|------------------------------------------|
| Driver           | 580.126.20    | CUDA 13.0 capable                        |
| torch wheel      | 2.11.0+cu128  | Pinned in pyproject.toml                 |
| torch.cuda       | 12.8          | What kernels are actually compiled for   |
| cuDNN            | 9.19          | Bundled with the cu128 wheel             |
| Target compute   | sm_100 (B200) | Reached via PTX JIT from cu128 build     |

**Why not upgrade right now:**
- PyTorch stable hasn't shipped cu130 wheels (as of 2026-05). The `cu128`
  pin in `pyproject.toml` is current-stable.
- Forward-compatibility: cu128 kernels run fine on driver 580 / CUDA 13.0.
- JIT path covers sm_100. First kernel launch on a new shape takes ~30s;
  cached after that. Not a sustained perf issue.
- Risk: untested cu130 + `torch.compile` + bf16 numerics regressions
  surface only under load. Find them in a smoke run, not a 60h run.

**When to revisit:** after the production run completes, OR if PyTorch
ships cu130 wheels with sm_100 binaries (no JIT) before then. Validate
on the smoke config first.

## Gotchas + diagnostics

| Issue | Fix |
|---|---|
| `--compile` first step takes 5+ min | Normal — Inductor caches kernels per shape; subsequent steps are instant |
| OOM at bs=96 | Drop to 64 in `data.loader.batch_size`, or set `memory_format: contiguous` |
| Slow data loading (GPU util <80%) | Check `iostat -x 2`; bump `num_workers` 24 → 32 |
| `_skipped_grad` shows up in logs | Single NaN gradient — fine. If >5/min, lower lr or amp=fp32 |
| Training stalls (no step movement 60s+) | `nvidia-smi`: util=0 → data starved; util=100% → check `_skipped_grad` rate |
| HF 401/403 on download | Re-run `huggingface-cli login` + request access on the dataset page |
| `--video-root` not present | Run prep_video_dataset.py first (step 4) |

## What's NOT auto-handled

- **NSFW filtering**: LAION-aesthetic has `punsafe<0.5` upstream filter,
  not perfect. For production deployment, add a downstream filter.
- **Disk monitoring**: training writes ~700 MB checkpoints every 10k steps;
  500k-step schedule = ~35 GB total under `runs/`. Plan accordingly.
- **HF token rotation**: tokens expire. If a download fails 401 a week in,
  re-login.

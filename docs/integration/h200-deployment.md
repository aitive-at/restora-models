# H200 SXM deployment guide

Target hardware (verified 2026-05-14):
- NVIDIA H200 (Hopper sm_90, ~143 GB HBM3e, 700 W TDP)
- Driver `570.211.01` (CUDA 12.8 ready)
- 20 CPU cores
- 1-10 Gbps internet (for LAION download)
- `/workspace/data` is the persistent data mount
- SSH-only access

The H200 is Hopper-architecture — same compute capability as H100, just
with more memory + faster HBM3e. PyTorch built for cu121 / cu124 / cu126
all work. The deployment commands below use cu126 (most stable as of
mid-2026); cu128 also works if available.

End-to-end from cold checkout to live training in **5 commands**.

## 1. Bootstrap

```sh
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone + sync
git clone https://github.com/aitive-at/restora-models.git
cd restora-models
uv sync

# Sanity check: GPU visible to torch?
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0))"
# Expected: cuda: True - NVIDIA H200 ...
```

If `cuda: False` here, fix it before going further (torch CPU-only wheel got
installed). Force the CUDA build:

```sh
uv pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch torchvision
```

## 2. HuggingFace auth

```sh
uv run huggingface-cli login
# Paste your token (https://huggingface.co/settings/tokens)
```

Then request access on the dataset page(s) you intend to use; approval is
usually instant:
- <https://huggingface.co/datasets/laion/relaion2B-multi-aesthetic> (17M, multi-language)
- <https://huggingface.co/datasets/laion/laion2B-en-aesthetic> (51M, English)
- <https://huggingface.co/datasets/laion/relaion1B-nolang-aesthetic> (52M, no language tag)

## 3. Download data → `/workspace/data`

```sh
# 17M multi-language images (~430 GB on disk at 384px). For an initial run,
# limit to a subset with --max-shards.
mkdir -p /workspace/data
uv run restora download \
  --output /workspace/data \
  --dataset relaion2B-multi-aesthetic \
  --processes 20 --threads 80
```

Re-running with the same flags resumes — both the parquet step and the
img2dataset image step skip work that's already complete. Safe to Ctrl-C
and restart any time.

If you want to start training before all 17M images are present, just
`--max-shards 8` (≈1M images / 40 GB) and start small.

## 4. Build the dataset manifest (one-time, fast)

```sh
uv run restora scan-data --root /workspace/data/images/relaion2B-multi-aesthetic
```

Writes `.restora-manifest.txt` so training startup doesn't have to walk
the tree from scratch.

## 5. Start training — background, SSH-disconnect safe

The training UI uses Rich Live which needs a TTY. The recommended pattern
is **tmux**: detach safely, reattach later, view live progress.

```sh
# One-time: install tmux if not already on the box
sudo apt-get install -y tmux       # or: pkg install / yum install / etc.
```

### NAFNet-large (recommended starting point — faster iteration)

```sh
# Start a named tmux session and launch training in it
tmux new-session -d -s train 'cd ~/restora-models && \
  uv run restora train \
    --config configs/h200-nafnet-large.yaml \
    --data /workspace/data/images/relaion2B-multi-aesthetic \
    --compile'
```

That's it. Training runs detached from your SSH session. Disconnect SSH,
go for coffee, your training keeps going.

### Reconnect + watch the Rich live dashboard

```sh
tmux attach -t train
# Press Ctrl+B then D to detach again (training keeps running)
```

The Rich UI shows: step counter / per-task PSNR / loss EMAs / GPU temp+mem+util.

### PromptIR-large alternative

```sh
tmux new-session -d -s train-promptir 'cd ~/restora-models && \
  uv run restora train \
    --config configs/h200-promptir-large.yaml \
    --data /workspace/data/images/relaion2B-multi-aesthetic \
    --compile'
tmux attach -t train-promptir
```

### Without tmux — fallback to nohup

If tmux isn't available:

```sh
cd ~/restora-models
nohup uv run restora train \
  --config configs/h200-nafnet-large.yaml \
  --data /workspace/data/images/relaion2B-multi-aesthetic \
  --compile \
  > /workspace/data/train.log 2>&1 &
disown
```

The Rich UI doesn't render in non-TTY, but periodic plaintext log lines
do go to stdout. Watch progress with:

```sh
tail -f /workspace/data/train.log
```

## 6. Export ONNX after training

```sh
# Per-task ONNX (RGB in, RGB out) — what most consumers want
for task in colorize denoise sharpen dejpeg deblur; do
  uv run restora export \
    --model runs/<run-dir>/ckpt/final.pt \
    --output ${task}.onnx \
    --task $task \
    --precision fp16
done

# Or one generic ONNX with config input
uv run restora export \
  --model runs/<run-dir>/ckpt/final.pt \
  --output model.onnx \
  --precision fp16
```

## Notes on the H200 config

The `h200-*.yaml` configs are tuned for the specific H200 hardware:
- batch_size **96** (NAFNet-large) / **40** (PromptIR-large) — ~8-5x the
  workstation defaults
- num_workers **20** — matches the CPU count
- lr scaled with batch (2e-4 for NAFNet, 1.5e-4 for PromptIR — PromptIR
  is more LR-sensitive)
- bf16 + `--compile` — both first-class on Hopper
- axis_probs.colorize **0.65** + chroma_lab weight **0.15** — slightly
  stronger color emphasis than the standard preset, feasible at production
  model size

If you find colorize still lagging after ~50k steps, bump chroma_lab
weight to 0.20-0.25 (edit the YAML in place — training is restartable).

## Gotchas + diagnostics

| Issue | Fix |
|---|---|
| `cuda: False` after `uv sync` | Force-reinstall CUDA torch (see step 1) |
| `--compile` first-step takes 5+ min | Normal — Inductor caches kernels per shape; subsequent runs at the same shape are instant |
| OOM at batch 96 | Drop to 64 or 48; or reduce `data.loader.prefetch_factor` |
| Slow data loading (img/s low, GPU util <80%) | Bump `num_workers` further (24-32) or check disk IO with `iostat -x 2` |
| Training stalls (no step counter movement for 60s) | Check `nvidia-smi`: if GPU util is 0, dataloader is starved. If util is 100% but no progress, gradient is being skipped (see logs for `_skipped_grad`) |
| HF 401 / 403 on download | Re-run `huggingface-cli login` + request access on the dataset page |
| `tmux: command not found` | Use the nohup fallback in step 5 |

## What's NOT auto-handled

- **NSFW filtering**: img2dataset writes whatever URLs the parquet lists.
  LAION-aesthetic has `punsafe<0.5` upstream filter, but it's not perfect.
  For production use, add a downstream filter.
- **Disk monitoring**: training writes checkpoints (~700 MB each every
  5000 steps) under `runs/`. With the 500k-step schedule that's ~70 GB.
  Plan disk accordingly.
- **HF token rotation**: tokens expire. If a download fails with 401 a
  week into a long run, re-login.

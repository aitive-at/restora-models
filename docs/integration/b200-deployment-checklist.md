# B200 cloud deployment checklist

Goal: `git clone` → `uv sync` → run training on a fresh B200 cloud machine
with no manual fiddling.

## Hardware assumptions

- NVIDIA B200 or B100 (Blackwell, sm_100, ~192 GB HBM)
- Modern Linux (Ubuntu 22.04+ or similar)
- 1-10 Gbps internet, plenty of NVMe local storage (≥ 2 TB recommended)
- CUDA driver 560+ (Blackwell requires recent driver)

## 1. Base setup (one-time, ~5 min)

```sh
# Install uv (single binary, no python required)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reload shell or: source ~/.bashrc

# Clone + sync
git clone <your-repo-url> coliraz
cd coliraz
uv sync                    # installs torch, opencv, img2dataset, etc.

# Verify CUDA visibility
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True NVIDIA B200 ...
```

If the CUDA check fails:
- Verify `nvidia-smi` works (driver installed)
- Verify the torch wheel has cu126 (or newer) suffix:
  `uv run python -c "import torch; print(torch.version.cuda)"` should return `12.6+`
- If torch is CPU-only, force-reinstall: `uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126`

## 2. HuggingFace auth (one-time)

```sh
# Either: interactive login (preferred — token saved to ~/.cache/huggingface/token)
uv run huggingface-cli login

# Or: env var
export HF_TOKEN=hf_xxx_your_token
# Add to ~/.bashrc for persistence
```

Request access on each dataset page you intend to use (see
`docs/integration/laion-download.md`). Approval is usually instant.

## 3. Download data (~10 min metadata, hours-to-days for images)

```sh
# Metadata-only first — fast, lets you sanity-check disk + auth
uv run refine download \
  --output /mnt/data/laion \
  --dataset relaion2B-multi-aesthetic \
  --max-shards 8 \
  --skip-images
# Verify: ls /mnt/data/laion/metadata/relaion2B-multi-aesthetic/   # 8 parquet files

# Full image download (background — disconnect-safe with nohup)
nohup uv run refine download \
  --output /mnt/data/laion \
  --dataset relaion2B-multi-aesthetic \
  --processes 32 --threads 128 \
  > download.log 2>&1 &
disown
```

Re-running with the same flags resumes; both metadata and image steps
skip completed shards.

## 4. Manifest the data (one-time after download)

```sh
uv run refine scan-data --root /mnt/data/laion/images/relaion2B-multi-aesthetic
# Writes .refine-manifest.txt; speeds up training startup
```

## 5. Train

```sh
# NAFNet-large production recipe (proven balanced — see commit d1b13f8)
uv run refine train \
  --config configs/laion-compound.yaml \
  --data /mnt/data/laion/images/relaion2B-multi-aesthetic \
  --compile

# PromptIR-large variant
uv run refine train \
  --config configs/promptir-laion.yaml \
  --data /mnt/data/laion/images/relaion2B-multi-aesthetic \
  --compile
```

`--compile` (torch.compile / Inductor) is the single biggest throughput win
on Blackwell. The EMA + checkpoint code already handles compiled models
correctly (commit 0d75778).

## 6. Export ONNX for deployment

```sh
# Single multi-task ONNX (input + config -> output)
uv run refine export --model runs/<run>/ckpt/final.pt --output model.onnx

# Per-task ONNX with config baked in (RGB in, RGB out)
for task in colorize denoise sharpen dejpeg deblur; do
  uv run refine export \
    --model runs/<run>/ckpt/final.pt \
    --output ${task}.onnx \
    --task $task
done

# fp16 quantization for inference servers
uv run refine export --model runs/<run>/ckpt/final.pt \
                     --output model_fp16.onnx --precision fp16
```

## Pre-flight gotchas

| Issue | Fix |
|---|---|
| `nvidia-ml-py` doesn't recognize B200 | Update to latest: `uv add 'nvidia-ml-py>=12.560'`. The training UI's GPU panel may show "unknown"; everything still works. |
| `--compile` first-step takes 5+ min | Normal — Inductor caches kernels per shape. Subsequent runs at the same shape are instant. |
| `fused=True` AdamW errors on CPU | Set in the YAML: `optim_g.fused: false` for CPU smoke tests; keep `true` for GPU. |
| OOM at batch 12 on B200 | Should not happen at 256² — B200 has 192 GB. If it does, suspect a memory leak (check no zombie eval batches piling up). |
| `bf16` autocast slow | Verify `torch.cuda.get_device_capability(0) == (10, 0)` — Blackwell has full bf16 tensor cores. |
| Slow data loading | Bump `data.loader.num_workers` to 32+ and `prefetch_factor` to 8. Cloud NVMe usually allows it. |
| fp8 export fails | Need `onnxruntime>=1.17` with cu126 build. The exporter emits a clear capability-error message if missing. |

## Disk planning

For full-scale production training:

- `/mnt/data/laion/` for downloaded data — plan 2-3 TB minimum, more if you'd
  like multiple datasets resident.
- `runs/` (the training output) — checkpoints are ~150 MB (tiny) to ~700 MB
  (large) each; with ema + last + final, a single run is ~2-3 GB. Plan 50 GB
  for `runs/`.
- ONNX exports — small (~30 MB tiny / ~150 MB large) per task.

## What's NOT auto-installed

The `uv sync` covers everything in `pyproject.toml`. Specifically NOT
included (deliberately):

- **Nothing system-level** — no apt packages, no compilers. Everything's
  via uv-managed Python wheels.
- **No checkpoints** — you train from scratch. If you want to pre-load a
  pretrained model, `scp` the `.pt` file in and point `--resume` at it.
- **No HuggingFace token** — auth is per-user; see step 2 above.

## Verification

```sh
# These should all succeed on a fresh B200 box after `uv sync`:
uv run pytest tests/ -q                              # 141 passed, 7 skipped
REFINE_SLOW=1 uv run pytest tests/ -q                # 148 passed
uv run refine --help                                 # CLI registered
uv run refine train --help                           # train command resolves
uv run refine download --help                        # download command resolves
uv run python -c "import torch; assert torch.cuda.is_available()"
```

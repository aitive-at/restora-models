# Cloud training runbook

Shared workflow for any of the supported rented GPUs. The only thing
that changes between hardware is which config you pass to `restora
train` — the training schedule, LR, model, and data layout are
identical across all three (see `configs/temporal-base.yaml`).

| GPU       | Config                            | Batch | Workers | Target VRAM |
|-----------|-----------------------------------|-------|---------|-------------|
| H200 SXM  | `configs/h200-temporal.yaml`      | 48    | 12      | ~83 GB      |
| B200 SXM  | `configs/b200-temporal.yaml`      | 64    | 16      | ~110 GB     |
| B300 SXM  | `configs/b300-temporal.yaml`      | 96    | 20      | ~165 GB     |

Assumed paths on every server:

| Purpose          | Path                                |
|------------------|-------------------------------------|
| Code checkout    | `/workspace/code/restora-models`    |
| Datasets         | `/workspace/data`                   |
| Run output (TB)  | `/workspace/runs/<config-name>`     |

## 1. First-time setup (run once per rental session)

```bash
git clone <repo> /workspace/code/restora-models
cd /workspace/code/restora-models
uv sync --extra dev
bash scripts/prepare.sh
```

`prepare.sh` is hardware-agnostic. It downloads REDS (`train_sharp` +
`val_sharp` from the official Hugging Face mirror at
`huggingface.co/datasets/snah/REDS`), unpacks the archives into
`/workspace/data/reds/<split>/<seq>/`, synthesises 600 film-overlay
PNGs, and verifies the layout. It is re-runnable — every step skips
work that's already done.

If the upstream URLs ever change, override on the fly:

```bash
REDS_TRAIN_SHARP_URLS="https://..." \
REDS_VAL_SHARP_URLS="https://..." \
bash scripts/prepare.sh
```

Done when the script prints `prepare.sh done.` and reports >0 sequences
for both `train_sharp` and `val_sharp` plus 600 overlay PNGs.

## 2. Starting training + TensorBoard in tmux

Pick the config that matches your GPU, then export `CFG` so the rest of
the commands stay GPU-agnostic. Each command is a single line
(paste-safe — no multi-line constructs, no heredocs — to dodge the
known tmux paste fragility).

```bash
export CFG=configs/h200-temporal.yaml   # or b200-temporal.yaml / b300-temporal.yaml
```

The training and TensorBoard sessions use the same `run.name` baked
into the config, so the TB logdir derives automatically:

```bash
tmux new-session -d -s train -c /workspace/code/restora-models
```
```bash
tmux send-keys -t train "uv run restora train --config ${CFG}" Enter
```
```bash
RUN_NAME=$(awk '/^run:/{f=1; next} f && /name:/{gsub(/"|^[ ]+name:[ ]+/, ""); print; exit}' ${CFG})
```
```bash
tmux new-session -d -s tb -c /workspace/code/restora-models
```
```bash
tmux send-keys -t tb "uv run tensorboard --logdir /workspace/runs/${RUN_NAME}/tb --bind_all --port 6006" Enter
```

Attach to either session to watch:

```bash
tmux attach -t train      # detach with Ctrl-b d
```
```bash
tmux attach -t tb         # detach with Ctrl-b d
```

The processes keep running across ssh disconnects because tmux owns
them.

TensorBoard is bound to `0.0.0.0:6006` — port-forward from your
workstation:

```bash
ssh -L 6006:localhost:6006 user@<rental-host>
```

then open `http://localhost:6006`.

## 3. What changes per GPU and what doesn't

**Same across H200 / B200 / B300:** `total_steps=80000`, `lr=3e-4`,
warmup 1500, cosine schedule, model `temporal_restora_small`, AMP
`bf16`, `torch.compile` on, channels-last, both REDS splits weighted
4:0.5, save every 5000.

**Different:** `data.loader.batch_size`, `data.loader.num_workers`,
`run.name`. That's it.

This is a deliberate choice: in the same number of steps a smaller
batch sees fewer samples, so the H200 run *technically* trains on less
data than the B300 run — but for "almost the same model" that's good
enough. If you ever need true equivalence, override `--total-steps` on
the CLI to match samples-seen instead of steps-taken.

## 4. Health-check after compile warmup (~100 s in)

Per `b200-deployment-2026-05-18` memory:

1. `runs/<config-name>/tb/events.out.*` exists and is >0 bytes. The
   6 Hz dashboard refresh (commit `753ec7f`) makes this visible without
   waiting for the first `log_every` tick.
2. Steady-state GPU util ≥ 85% in the rich-live dashboard's GPU panel.
   If it sits at ~60%, the dataloader is the bottleneck — bump
   `num_workers` toward the box's core count.
3. Steady-state VRAM near the table's target. If it's significantly
   higher than expected, drop batch_size one notch; if much lower,
   you have headroom to bump.
4. Total-loss curve monotone post-warmup; PSNR climbing by ~step 5k.
   If PSNR oscillates in the first 5k, drop `lr` to 2.5e-4 in the
   shared `configs/temporal-base.yaml`.

## 5. Stopping / cleaning up

```bash
tmux kill-session -t train
```
```bash
tmux kill-session -t tb
```

Latest weights survive in `/workspace/runs/<config-name>/` —
`final.pt` once a full run completes, otherwise the most recent
`iter_NNNNNNN.pt` checkpoint (cadence: every 5 000 steps).

## 6. Exporting the trained model to ONNX

The CLI defaults to EMA weights (usually higher-quality than live
training weights — typically 1-3 dB PSNR better on REDS):

```bash
uv run restora export --model /workspace/runs/${RUN_NAME}/final.pt --output /workspace/exports/${RUN_NAME}_all.onnx --task all --precision fp16
```

`--task all` bakes the 5-axis config as a constant in the graph (the
ONNX has a single `frames` input). Use `--task colorize|denoise|sharpen|dejpeg|deblur`
for single-axis ONNX files, or omit `--task` for the generic 2-input
form. Pass `--weights model` if you specifically want the live training
weights instead of EMA.

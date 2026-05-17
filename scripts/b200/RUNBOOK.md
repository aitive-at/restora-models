# B200 server runbook

Everything in this directory targets the **rented B200 box** (180 GB
VRAM, 28 CPU cores, 280 GB RAM) — keep it mentally separate from the
local-Blackwell flow under `configs/local-temporal.yaml`.

Assumed paths on the server:

| Purpose          | Path                                |
|------------------|-------------------------------------|
| Code checkout    | `/workspace/code/restora-models`    |
| Datasets         | `/workspace/data`                   |
| Run output (TB)  | `/workspace/runs/b200_temporal`     |

## 1. First-time setup (run once per rental session)

```bash
git clone <repo> /workspace/code/restora-models
cd /workspace/code/restora-models
uv sync --extra dev
bash scripts/b200/prepare.sh
```

`prepare.sh` downloads REDS (`train_sharp` + `val_sharp` from the
official Hugging Face mirror at `huggingface.co/datasets/snah/REDS`),
unpacks the archives into `/workspace/data/reds/<split>/<seq>/`,
synthesises 600 film-overlay PNGs, and verifies the layout. It's
re-runnable — every step skips work that's already done.

If the upstream URLs ever change, override on the fly:

```bash
REDS_TRAIN_SHARP_URLS="https://..." \
REDS_VAL_SHARP_URLS="https://..." \
bash scripts/b200/prepare.sh
```

Done when the script prints `prepare.sh done.` and reports >0 sequences
for both `train_sharp` and `val_sharp` plus 600 overlay PNGs.

## 2. Starting training + TensorBoard in tmux

Each command is a single line (paste-safe — no multi-line constructs,
no heredocs — to dodge the known tmux paste fragility).

```bash
tmux new-session -d -s train -c /workspace/code/restora-models
```
```bash
tmux send-keys -t train 'uv run restora train --config configs/b200-temporal.yaml' Enter
```
```bash
tmux new-session -d -s tb -c /workspace/code/restora-models
```
```bash
tmux send-keys -t tb 'uv run tensorboard --logdir /workspace/runs/b200_temporal/tb --bind_all --port 6006' Enter
```

Attach to either session to watch:

```bash
tmux attach -t train
```
```bash
tmux attach -t tb
```

Detach with `Ctrl-b d`. The processes keep running across ssh
disconnects because tmux owns them.

TensorBoard is bound to `0.0.0.0:6006` — port-forward from your
workstation:

```bash
ssh -L 6006:localhost:6006 user@b200-host
```
then open `http://localhost:6006`.

## 3. Why the B200 config differs from the local one

| Knob              | local-temporal.yaml | b200-temporal.yaml | Why                                                      |
|-------------------|---------------------|--------------------|----------------------------------------------------------|
| `batch_size`      | 32                  | 64                 | ~110 GB at bs=64 vs 180 GB available; 70 GB headroom     |
| `num_workers`     | 12                  | 16                 | 28 cores, degradation is CPU-bound, RAM plentiful        |
| `lr`              | 2e-4                | 3e-4               | Larger batch → can push harder; full 4e-4 felt risky     |
| `warmup_steps`    | 1000                | 1500               | Longer ramp for the higher peak LR                       |
| `total_steps`     | 60 000              | 80 000             | ~5.1 M samples vs 3.8 M — uses the extra compute         |
| `run.root`        | `runs/` (default)   | `/workspace/runs`  | Stay off the small system disk                           |
| Data paths        | `~/data/...`        | `/workspace/data/` | Container/cloud convention                                |

Override on the CLI without editing the config:

```bash
uv run restora train --config configs/b200-temporal.yaml --total-steps 100000 --batch-size 80
```

**Training data:** both `train_sharp` (240 sequences) and `val_sharp`
(30 sequences) are mixed into the training pool, weighted 4.0:0.5
(roughly the 8:1 sequence-count ratio). To hold val_sharp out for honest
evaluation instead, delete its source entry from `b200-temporal.yaml`.

## 4. Stopping / cleaning up

```bash
tmux kill-session -t train
```
```bash
tmux kill-session -t tb
```

The latest weights survive in `/workspace/runs/b200_temporal/` —
`final.pt` once a full run completes, otherwise the most recent
`iter_NNNNNNN.pt` checkpoint (cadence: every 5 000 steps).

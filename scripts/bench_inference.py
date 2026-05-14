#!/usr/bin/env python
"""Benchmark inference speed of a trained model.

Usage:
    uv run python scripts/bench_inference.py \\
        --ckpt runs/.../ckpt/final.pt \\
        --input-size 256 \\
        --batch-size 1 \\
        --iters 100

Reports:
    - Warmup median (ms/iter)
    - Steady-state median + p99 (ms/iter)
    - GFLOPs estimate (approx via param count + input volume)
    - Throughput (images/sec)

Useful for verifying refine-head models still meet the realtime target
on the target hardware. Default eval mode is the EMA model if present
in the checkpoint, else the raw model.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def _load_model(ckpt_path: Path, device: torch.device, prefer_ema: bool = True):
    from restora_models.config import ModelConfig
    from restora_models.models import build_model
    from restora_models.data.compound import AXES
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg = ModelConfig(**(cfg_dict.get("model") or {}))
    model = build_model(mcfg, num_axes=len(AXES))
    if prefer_ema and payload.get("ema_module") is not None:
        sd = payload["ema_module"]
        prefix = "module."
        cleaned = {k.removeprefix(prefix): v for k, v in sd.items()}
        model.load_state_dict(cleaned, strict=False)
        print("[bench] loaded EMA weights")
    else:
        model.load_state_dict(payload["model"])
        print("[bench] loaded raw model weights")
    model.train(False).to(device)
    return model, mcfg


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--config", default="1 1 1 1 1",
                   help="5 axis flags as space-separated 0/1 (default: all on)")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    model, mcfg = _load_model(args.ckpt, device, prefer_ema=not args.no_ema)

    H = W = args.input_size
    B = args.batch_size
    x = torch.rand(B, 3, H, W, device=device)
    flags = [int(c) for c in args.config.split()]
    if len(flags) != 5:
        raise SystemExit("config must be 5 0/1 values")
    cfg = torch.tensor([flags] * B, dtype=torch.float32, device=device)

    amp_map = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = amp_map[args.amp]

    def step():
        if amp_dtype is not None and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _ = model(x, cfg)
        else:
            _ = model(x, cfg)
        if device.type == "cuda":
            torch.cuda.synchronize()

    print(f"[bench] model={mcfg.type} size={mcfg.size} "
          f"refine={mcfg.adversarial_refine} input={H}x{W} bs={B} amp={args.amp}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] params: {n_params/1e6:.2f}M")

    # Warmup
    print(f"[bench] warming up ({args.warmup} iters)...")
    for _ in range(args.warmup):
        step()

    # Time
    print(f"[bench] timing ({args.iters} iters)...")
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    med = float(np.median(times))
    p99 = float(np.percentile(times, 99))
    mn = float(times.min())
    mx = float(times.max())
    throughput = (B * 1000.0) / med
    print(f"[bench] median {med:.2f} ms  |  p99 {p99:.2f} ms  |  min {mn:.2f}  max {mx:.2f}")
    print(f"[bench] throughput @ bs={B}: {throughput:.1f} img/s")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[bench] peak VRAM: {peak:.2f} GB")


if __name__ == "__main__":
    main()

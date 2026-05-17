"""Inference benchmark for a temporal checkpoint."""
from __future__ import annotations

import time
from pathlib import Path
from statistics import median, quantiles

import torch

from restora_models.config import ModelConfig
from restora_models.data.compound import AXES
from restora_models.models.registry import build_model


def run_bench(
    *,
    ckpt: Path,
    input_size: int = 256,
    batch_size: int = 1,
    iters: int = 100,
    warmup: int = 10,
    device: str | None = None,
    amp: str = "bf16",
    config_axes: tuple[int, ...] = (1, 1, 1, 1, 1),
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mtype = (cfg_dict.get("model") or {}).get("type", "temporal_restora_small")
    m = build_model(ModelConfig(type=mtype), num_axes=len(AXES))
    m.train(False)
    m = m.to(dev)
    m.load_state_dict(payload["model"])

    if dev.type == "cuda":
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp]
    else:
        amp_dtype = torch.float32

    frames = torch.rand(batch_size, 7, 3, input_size, input_size, device=dev)
    config = torch.tensor([list(config_axes)] * batch_size, dtype=torch.float32, device=dev)

    # Warmup
    with torch.inference_mode(), torch.amp.autocast(dev.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        for _ in range(warmup):
            _ = m(frames, config)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    times = []
    with torch.inference_mode(), torch.amp.autocast(dev.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        for _ in range(iters):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = m(frames, config)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    median_s = median(times)
    p99_s = quantiles(times, n=100)[98]
    fps = batch_size / median_s
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else 0.0
    print(f"input={input_size}x{input_size} bs={batch_size} amp={amp}")
    print(f"  median: {median_s * 1000:.2f} ms  ({fps:.1f} fps)")
    print(f"  p99:    {p99_s * 1000:.2f} ms")
    if peak_vram_mb:
        print(f"  peak VRAM: {peak_vram_mb:.1f} MB")
    return {"median_ms": median_s * 1000, "p99_ms": p99_s * 1000, "fps": fps, "peak_vram_mb": peak_vram_mb}

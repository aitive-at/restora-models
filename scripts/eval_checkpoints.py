#!/usr/bin/env python
"""Compare per-task PSNR for two checkpoints on a fixed validation set.

Builds a deterministic validation batch by sampling N images from the
image root, applying each restoration axis as a separate one-hot config,
and measuring PSNR(pred, clean) for each axis under each checkpoint.
Reports delta between the two checkpoints.

Usage:
    uv run python scripts/eval_checkpoints.py \\
        --ckpts runs/.../ckpt/last.pt runs/.../ckpt/final.pt \\
        --data ~/data/laion-images \\
        --n 32 --seed 0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from restora_models.config import ModelConfig
from restora_models.data.compound import AXES, DEGRADE_ORDER, AXIS_TO_REG
from restora_models.data.dataset import RecursiveImageDataset
from restora_models.data.degradations.registry import build_degradation
from restora_models.data.degradations import (
    colorization as _c, deblur as _d, denoise as _dn,
    jpeg as _j, superres as _s,
)
from restora_models.losses.metrics import psnr
from restora_models.models import build_model


def _load_model(ckpt_path: Path, device: torch.device):
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**((payload.get("extra") or {}).get("cfg") or {}).get("model", {}))
    model = build_model(mcfg, num_axes=len(AXES)).to(device)
    model.load_state_dict(payload["model"])
    model.train(False)
    return model, payload.get("step", 0)


def _build_eval_batch(image_root: Path, n: int, target_size: int,
                       device: torch.device, seed: int):
    """Build a (clean, degraded_per_axis) batch for evaluation.

    Returns dict: {axis_name: (clean_tensor, degraded_tensor) (B,3,H,W)}
    Each axis sees the SAME clean images with the corresponding single
    degradation applied.
    """
    import random
    ds = RecursiveImageDataset(image_root, target_size=target_size,
                                val_fraction=0.0, split="all",
                                augment_hflip=False, seed=seed)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    cleans = torch.stack([ds[i] for i in idxs]).to(device)   # (B, 3, H, W)

    # Build axis-specific degraded batches
    out = {}
    deg_params = {
        "colorize": {},
        "denoise":  {"sigma_range": [0.03, 0.03]},   # fixed σ for stable eval
        "sharpen":  {"factor_choices": [4]},         # fixed 4x SR
        "dejpeg":   {"quality_range": [40, 40]},     # fixed Q40
        "deblur":   {"sigma_range": [2.0, 2.0], "motion_prob": 0.0},
    }
    for axis in AXES:
        deg = build_degradation(AXIS_TO_REG[axis], deg_params[axis])
        batch_deg = []
        for i, idx in enumerate(idxs):
            rng_i = random.Random(seed * 10000 + i)
            rgb = ds[idx].permute(1, 2, 0).numpy().copy()
            rgb = deg.degrade(rgb, rng_i)
            batch_deg.append(torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous())
        out[axis] = (cleans, torch.stack(batch_deg).to(device))
    return out


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpts", nargs="+", type=Path, required=True,
                   help="One or more checkpoints to compare")
    p.add_argument("--data", type=Path, default=Path("~/data/laion-images"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    data_root = args.data.expanduser()
    print(f"[eval] building eval batch from {data_root}: n={args.n} size={args.input_size}")
    batches = _build_eval_batch(data_root, args.n, args.input_size, device, args.seed)

    # Eval each checkpoint
    results: dict[Path, dict[str, float]] = {}
    for ckpt in args.ckpts:
        model, step = _load_model(ckpt, device)
        print(f"\n[eval] {ckpt.name}  step={step}")
        ckpt_results = {}
        for axis, (clean, degraded) in batches.items():
            cfg_vec = [0.0] * 5
            cfg_vec[list(AXES).index(axis)] = 1.0
            cfg = torch.tensor([cfg_vec] * clean.shape[0],
                                dtype=torch.float32, device=device)
            pred = model(degraded, cfg).clamp(0, 1)
            per_sample = psnr(pred, clean)
            mean_psnr = float(per_sample.mean().item())
            ckpt_results[axis] = mean_psnr
            print(f"  {axis:>10s}: {mean_psnr:6.2f} dB")
        results[ckpt] = ckpt_results

    # Delta vs first
    if len(args.ckpts) > 1:
        base = args.ckpts[0]
        print(f"\n[eval] delta vs {base.name}:")
        for ckpt in args.ckpts[1:]:
            print(f"  {ckpt.name}:")
            for axis in AXES:
                d = results[ckpt][axis] - results[base][axis]
                arrow = "▲" if d > 0 else ("▼" if d < 0 else "·")
                print(f"    {axis:>10s}: {d:+6.3f} dB {arrow}")


if __name__ == "__main__":
    main()

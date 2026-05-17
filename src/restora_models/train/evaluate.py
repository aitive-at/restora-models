"""Per-axis PSNR comparison across one or more checkpoints."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import torch

from restora_models.config import ModelConfig
from restora_models.data.compound import AXES
from restora_models.data.reds import REDSDataset
from restora_models.models.registry import build_model
from restora_models.train.trainer import _apply_per_frame_degradations, _build_per_frame_degradations


def _eval_ckpt_psnr(ckpt: Path, *, holdout, device: torch.device, per_frame_degs: dict) -> dict[str, float]:
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mtype = (cfg_dict.get("model") or {}).get("type", "temporal_restora_small")
    m = build_model(ModelConfig(type=mtype), num_axes=len(AXES))
    m.train(False)
    m = m.to(device)
    m.load_state_dict(payload["model"])
    tests = {**{ax: {ax} for ax in AXES}, "all": set(AXES)}
    results = {}
    with torch.inference_mode():
        for label, axes in tests.items():
            psnrs = []
            for sample in holdout:
                clean_clip = sample["frames"]
                cfgvec = torch.zeros(1, len(AXES), device=device)
                for i, a in enumerate(AXES):
                    if a in axes:
                        cfgvec[0, i] = 1.0
                rng = random.Random(123)
                deg = _apply_per_frame_degradations(clean_clip, axes, per_frame_degs, rng) if axes else clean_clip
                pred = m(deg.unsqueeze(0).to(device), cfgvec)
                target = clean_clip[3].unsqueeze(0).to(device)
                mse = ((pred - target) ** 2).mean().item()
                psnrs.append(-10 * math.log10(max(mse, 1e-9)))
            results[label] = sum(psnrs) / len(psnrs)
    return results


def run_compare(*, ckpts: Sequence[Path], data: Path, n: int = 16, seed: int = 0, device: str | None = None) -> None:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = REDSDataset(data, split="train_sharp", window=7, stride=1, crop=256)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    holdout = [ds[i] for i in idxs]
    per_frame_degs = _build_per_frame_degradations()

    rows = []
    for ck in ckpts:
        rows.append((ck, _eval_ckpt_psnr(ck, holdout=holdout, device=dev, per_frame_degs=per_frame_degs)))

    if not rows:
        return
    axes = list(rows[0][1].keys())
    header = "Checkpoint".ljust(40) + "  " + "  ".join(f"{a:>8s}" for a in axes)
    print(header)
    print("-" * len(header))
    base = rows[0][1]
    for ck, scores in rows:
        line = str(ck)[-40:].ljust(40)
        for a in axes:
            v = scores[a]
            line += f"  {v:8.2f}"
        print(line)
    if len(rows) > 1:
        print("--- deltas vs first checkpoint ---")
        for ck, scores in rows[1:]:
            line = str(ck)[-40:].ljust(40)
            for a in axes:
                d = scores[a] - base[a]
                line += f"  {d:+8.2f}"
            print(line)

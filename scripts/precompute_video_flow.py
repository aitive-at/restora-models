#!/usr/bin/env python
"""Precompute RAFT optical flow for all video frame pairs.

Walks `<root>/<video_name>/*.jpg` and produces, for each (t, t+k) pair
with k in [1, max_skip], a flow file at:

    <root>/<video_name>/.flow/frame_NNNNN_skipK.npz   (key: 'flow' shape (2,H,W))

Resumable: existing flow files are skipped on re-run.

Usage:
    uv run python scripts/precompute_video_flow.py \\
        --root ~/data/laion-videos \\
        --max-skip 5 \\
        --resolution 256
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def _load_image_as_tensor(path: Path, h: int, w: int, device: torch.device) -> torch.Tensor:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"cv2 failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Video root with one subdir per video")
    p.add_argument("--max-skip", type=int, default=5,
                   help="Max temporal skip k (compute flow for k=1..max_skip)")
    p.add_argument("--resolution", type=int, default=256,
                   help="Resolution to compute flow at (square). 256 is fine for "
                        "training at 256 squared. RAFT works at any size but is "
                        "fastest at lower res.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--print-every", type=int, default=20)
    args = p.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"{root} doesn't exist")

    device = torch.device(args.device)
    print(f"[raft] loading RAFT_Large to {device}...", flush=True)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
    model.train(False)
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    # Walk videos
    video_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not video_dirs:
        raise SystemExit(f"no video subdirs under {root}")

    print(f"[raft] found {len(video_dirs)} videos under {root}", flush=True)
    t0 = time.time()
    total_pairs = 0
    skipped = 0
    done = 0

    for vi, vd in enumerate(video_dirs):
        frames = sorted(
            p for p in vd.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(frames) < 2:
            continue
        flow_dir = vd / ".flow"
        flow_dir.mkdir(exist_ok=True)

        for fi in range(len(frames) - 1):
            max_k = min(args.max_skip, len(frames) - fi - 1)
            for k in range(1, max_k + 1):
                out_path = flow_dir / f"frame_{fi:05d}_skip{k}.npz"
                total_pairs += 1
                if out_path.exists() and out_path.stat().st_size > 0:
                    skipped += 1
                    continue
                img_t = _load_image_as_tensor(frames[fi], args.resolution,
                                              args.resolution, device)
                img_tk = _load_image_as_tensor(frames[fi + k], args.resolution,
                                               args.resolution, device)
                img_t_n, img_tk_n = transforms(img_t, img_tk)
                flow_list = model(img_t_n, img_tk_n)
                flow = flow_list[-1][0].cpu().numpy().astype(np.float32)
                np.savez_compressed(out_path, flow=flow)
                done += 1

        if (vi + 1) % args.print_every == 0 or vi == len(video_dirs) - 1:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-3)
            print(
                f"[raft] {vi + 1}/{len(video_dirs)} videos | "
                f"{done} flows computed, {skipped} skipped, "
                f"{rate:.1f} flows/s, {elapsed:.0f}s elapsed",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"[raft] done. {done} new flows / {skipped} skipped / {total_pairs} total "
        f"in {elapsed:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

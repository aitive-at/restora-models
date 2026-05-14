#!/usr/bin/env python
"""Generate synthetic video pairs from existing LAION images.

For each source image we synthesize N short 'videos' of K frames by
applying a smoothly-varying affine transform (random translation +
rotation per video). This gives us:

    out/<video_id>/frame_NNNNN.jpg

plus precomputed ground-truth optical flow (no RAFT needed — we KNOW the
transform):

    out/<video_id>/.flow/frame_NNNNN_skipK.npz  (key: 'flow')

The flow is computed analytically from the per-frame affine matrices.
This is a much cleaner training signal than RAFT-estimated flow because
we have zero estimation error.

This is a stopgap so we can validate the trainer's video path while
real video data (DAVIS-2017) is downloading.

Usage:
    uv run python scripts/make_synthetic_videos.py \\
        --source ~/data/laion-images \\
        --out ~/data/synthetic-videos \\
        --num-videos 200 \\
        --frames-per-video 5 \\
        --resolution 256
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def _iter_image_paths(root: Path, limit: int):
    out = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                out.append(f)
                if len(out) >= limit:
                    return out
    return out


def _affine_matrix(tx: float, ty: float, angle_deg: float, center: tuple[float, float]
                    ) -> np.ndarray:
    """2x3 affine: rotation about center + translation."""
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def _warp_with_affine(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def _affine_inv_2x3(M: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    M3 = np.vstack([M, [0, 0, 1]])
    Mi = np.linalg.inv(M3)
    return Mi[:2]


def _flow_between(M_t: np.ndarray, M_tk: np.ndarray, H: int, W: int
                   ) -> np.ndarray:
    """Backward flow from frame_t+k to frame_t — the kind flow_warp needs.

    flow_warp(image, flow) samples image at (p + flow[p]). For
    warp(frame_t, flow) ≈ frame_t+k we need: for each pixel p in
    frame_t+k's grid, where in frame_t lives the same physical content.

    Both frames are rendered from a shared source via affines:
        frame_t[q]  = source[M_t^{-1} q]
        frame_tk[p] = source[M_tk^{-1} p]
    Same content ↔ M_t^{-1} q = M_tk^{-1} p ↔ q = M_t M_tk^{-1} p.
    Therefore: flow[p] = q - p = (M_t M_tk^{-1} - I) p.
    """
    M_tk_inv = np.eye(3)
    M_tk_inv[:2] = _affine_inv_2x3(M_tk)
    M_t_full = np.eye(3)
    M_t_full[:2] = M_t
    comp = M_t_full @ M_tk_inv

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32), indexing="ij")
    ones = np.ones_like(xx)
    pts = np.stack([xx, yy, ones], axis=-1)
    pts_t = pts @ comp.T
    flow = pts_t[..., :2] - pts[..., :2]
    return flow.transpose(2, 0, 1).astype(np.float32)


def _sample_per_frame_affines(K: int, H: int, W: int, rng: random.Random):
    """Generate K affines that drift smoothly over time."""
    # Pick a per-video direction + speed; pick a per-video rotation rate.
    max_drift = 0.05 * min(H, W)    # up to 5% of dim per frame
    angle_max = 1.5                  # ±1.5 deg per frame
    dx = rng.uniform(-max_drift, max_drift)
    dy = rng.uniform(-max_drift, max_drift)
    da = rng.uniform(-angle_max, angle_max)
    cx, cy = W / 2.0, H / 2.0
    return [
        _affine_matrix(dx * t, dy * t, da * t, (cx, cy)) for t in range(K)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="Source image dataset root (LAION layout)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output root — videos will be created as subdirs")
    p.add_argument("--num-videos", type=int, default=200,
                   help="How many videos to generate")
    p.add_argument("--frames-per-video", type=int, default=5)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max-skip", type=int, default=5,
                   help="Precompute flow for skip = 1..max_skip")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    source = args.source.expanduser()
    out = args.out.expanduser()
    if not source.exists():
        raise SystemExit(f"{source} not found")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[synth] scanning {source} for source images...")
    paths = _iter_image_paths(source, args.num_videos * 2)
    if len(paths) < args.num_videos:
        raise SystemExit(f"only {len(paths)} source images found")
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: args.num_videos]

    H = W = args.resolution
    K = args.frames_per_video
    print(f"[synth] generating {args.num_videos} videos x {K} frames @ {H}x{W}")

    n_done = 0
    for vi, src_path in enumerate(paths):
        vid_id = f"vid_{vi:05d}"
        vid_dir = out / vid_id
        flow_dir = vid_dir / ".flow"
        if vid_dir.exists() and any(vid_dir.glob("frame_*.jpg")):
            continue
        vid_dir.mkdir(parents=True, exist_ok=True)
        flow_dir.mkdir(exist_ok=True)

        bgr = cv2.imread(str(src_path))
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)

        # Per-video affines
        per_frame_rng = random.Random((args.seed * 1_000_003) ^ vi)
        Ms = _sample_per_frame_affines(K, H, W, per_frame_rng)
        for ti, M in enumerate(Ms):
            warped = _warp_with_affine(bgr, M)
            cv2.imwrite(str(vid_dir / f"frame_{ti:05d}.jpg"), warped,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Precomputed analytical flows
        for ti in range(K - 1):
            max_k = min(args.max_skip, K - ti - 1)
            for k in range(1, max_k + 1):
                flow = _flow_between(Ms[ti], Ms[ti + k], H, W)
                np.savez_compressed(
                    flow_dir / f"frame_{ti:05d}_skip{k}.npz", flow=flow
                )
        n_done += 1
        if (vi + 1) % 20 == 0:
            print(f"[synth] {vi + 1}/{args.num_videos} videos written")

    print(f"[synth] done. {n_done} new videos under {out}")


if __name__ == "__main__":
    sys.exit(main())

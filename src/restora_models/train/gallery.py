"""Qualitative gallery: triptychs of (clean | degraded | restored) on the center frame."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch

from restora_models.config import ModelConfig
from restora_models.data.compound import AXES
from restora_models.data.reds import REDSDataset
from restora_models.models.registry import build_model
from restora_models.train.trainer import _apply_per_frame_degradations, _build_per_frame_degradations


def _triptych(clean: np.ndarray, degraded: np.ndarray, restored: np.ndarray) -> np.ndarray:
    h, w = clean.shape[:2]
    out = np.zeros((h, w * 3 + 8, 3), dtype=np.uint8)
    out[:, 0:w] = clean
    out[:, w + 4:2 * w + 4] = degraded
    out[:, 2 * w + 8:3 * w + 8] = restored
    return out


def run_gallery(
    *, ckpt: Path, data: Path, out: Path,
    n: int = 16, axis: str = "colorize", input_size: int = 256,
    seed: int = 0, device: str | None = None,
) -> None:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mtype = (cfg_dict.get("model") or {}).get("type", "temporal_restora_small")
    m = build_model(ModelConfig(type=mtype), num_axes=len(AXES))
    m.train(False)
    m = m.to(dev)
    m.load_state_dict(payload["model"])

    ds = REDSDataset(data, split="train_sharp", window=7, stride=1, crop=input_size)
    rng_pick = random.Random(seed)
    idxs = rng_pick.sample(range(len(ds)), min(n, len(ds)))
    per_frame_degs = _build_per_frame_degradations()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)

    cfg_vec = torch.zeros(1, len(AXES), device=dev)
    cfg_vec[0, AXES.index(axis)] = 1.0

    with torch.inference_mode():
        for i, idx in enumerate(idxs):
            sample = ds[idx]
            clean_clip = sample["frames"]
            rng = random.Random(idx)
            deg_clip = _apply_per_frame_degradations(clean_clip, {axis}, per_frame_degs, rng)
            pred = m(deg_clip.unsqueeze(0).to(dev), cfg_vec).clamp(0, 1).squeeze(0)
            center_clean = clean_clip[3]
            center_deg = deg_clip[3]
            def to_bgr(t):
                arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            tri = _triptych(to_bgr(center_clean), to_bgr(center_deg), to_bgr(pred))
            cv2.imwrite(str(out / f"sample_{i:03d}.png"), tri)

    print(f"wrote {len(idxs)} triptychs to {out}")

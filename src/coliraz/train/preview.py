"""Render preview comparison grid: [original | gray | pred | |delta_ab|]."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch


def _t_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).round().astype(np.uint8)


def _delta_ab_to_heatmap(delta_ab: torch.Tensor, max_val: float = 50.0) -> np.ndarray:
    mag = torch.linalg.vector_norm(delta_ab.detach().float().cpu(), dim=0)
    mag = (mag.clamp(0, max_val) / max_val * 255.0).to(torch.uint8).numpy()
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def render_preview_grid(
    samples: list[dict[str, torch.Tensor]],
    *,
    caption: str,
    cell_size: int = 256,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for s in samples:
        tiles = []
        for key in ("original", "gray_rgb", "pred_rgb"):
            img = _t_to_uint8_rgb(s[key])
            if img.shape[:2] != (cell_size, cell_size):
                img = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            tiles.append(img)
        delta = _delta_ab_to_heatmap(s["delta_ab"])
        if delta.shape[:2] != (cell_size, cell_size):
            delta = cv2.resize(delta, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        rows.append(np.concatenate(tiles + [delta], axis=1))
    body = np.concatenate(rows, axis=0)

    cap_h = 24
    cap = np.zeros((cap_h, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(cap, caption, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([cap, body], axis=0)


def write_png_atomic(path: str | Path, img_rgb_uint8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)

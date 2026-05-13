"""Multi-task preview grid renderer + atomic PNG writer."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch


def _t_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).round().astype(np.uint8)


def _delta_heatmap(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 0.5) -> np.ndarray:
    delta = (pred - clean).detach().float().cpu()
    mag = torch.linalg.vector_norm(delta, dim=0)
    mag = (mag.clamp(0, max_val) / max_val * 255.0).to(torch.uint8).numpy()
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def render_multitask_grid(
    samples: dict[str, list[dict[str, torch.Tensor]]],
    *,
    caption: str,
    cell_size: int = 256,
) -> np.ndarray:
    """samples: {task_name: [{"clean":..., "degraded":..., "predicted":...}, ...]}.

    Each task row contains all samples (concatenated horizontally) of:
    clean | degraded | predicted | |Δ| heatmap.
    Tasks stacked vertically. Top caption strip.
    """
    rows: list[np.ndarray] = []
    for task_name, sample_list in samples.items():
        per_sample_rows: list[np.ndarray] = []
        for s in sample_list:
            tiles = []
            for key in ("clean", "degraded", "predicted"):
                img = _t_to_uint8_rgb(s[key])
                if img.shape[:2] != (cell_size, cell_size):
                    img = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
                tiles.append(img)
            delta = _delta_heatmap(s["predicted"], s["clean"])
            if delta.shape[:2] != (cell_size, cell_size):
                delta = cv2.resize(delta, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            per_sample_rows.append(np.concatenate(tiles + [delta], axis=1))
        row = np.concatenate(per_sample_rows, axis=0)
        label = np.zeros((row.shape[0], 28, 3), dtype=np.uint8)
        cv2.putText(label, task_name[:10], (2, row.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([label, row], axis=1))

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

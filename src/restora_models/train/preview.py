"""Multi-task preview grid renderer + atomic PNG writer."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_COLOR = (255, 255, 255)
_LINE_TYPE = cv2.LINE_AA

# Labeling strip heights & font scales — tuned to read clearly at typical
# cell sizes (256 px training preview, 32 px in tests).
_GLOBAL_CAPTION_H = 32
_GLOBAL_CAPTION_SCALE = 0.6
_COL_HEADER_H = 28
_COL_HEADER_SCALE = 0.55
_TASK_HEADER_H = 30
_TASK_HEADER_SCALE = 0.65

_COLUMN_LABELS = ("Clean (GT)", "Degraded (input)", "Predicted", "|delta| heatmap")


def _t_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255.0).round().astype(np.uint8)


def _delta_heatmap(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 0.5) -> np.ndarray:
    delta = (pred - clean).detach().float().cpu()
    mag = torch.linalg.vector_norm(delta, dim=0)
    mag = (mag.clamp(0, max_val) / max_val * 255.0).to(torch.uint8).numpy()
    return cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)


def _draw_centered(canvas: np.ndarray, text: str, x_start: int, x_end: int,
                   scale: float, thickness: int = 1) -> None:
    """Draw text centered horizontally within [x_start, x_end), vertically centered in canvas."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    x = x_start + max(0, ((x_end - x_start) - tw) // 2)
    y = (canvas.shape[0] + th) // 2
    cv2.putText(canvas, text, (x, y), _FONT, scale, _TEXT_COLOR, thickness, _LINE_TYPE)


def _draw_left(canvas: np.ndarray, text: str, x: int, scale: float, thickness: int = 1) -> None:
    (_, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    y = (canvas.shape[0] + th) // 2
    cv2.putText(canvas, text, (x, y), _FONT, scale, _TEXT_COLOR, thickness, _LINE_TYPE)


def render_multitask_grid(
    samples: dict[str, list[dict[str, torch.Tensor]]],
    *,
    caption: str,
    cell_size: int = 256,
) -> np.ndarray:
    """samples: {task_name: [{"clean":..., "degraded":..., "predicted":...}, ...]}.

    Layout (top to bottom):
      [global caption strip — step + timestamp]
      for each task:
        [blank spacer (8 px)]
        [task caption strip — full-width, names the task]
        [column header — Clean | Degraded | Predicted | |delta| heatmap]
        [task's samples — one per row, four columns each]

    Repeating the column header per task means every row of images is
    fully self-described (task + per-column meaning), with no need to
    scroll to the top of the preview to remember what each column is.

    Each cell is `cell_size x cell_size`; the grid is 4 * cell_size wide.
    """
    total_width = 4 * cell_size
    rows: list[np.ndarray] = []

    # Global caption strip: step + timestamp
    gc = np.zeros((_GLOBAL_CAPTION_H, total_width, 3), dtype=np.uint8)
    _draw_left(gc, caption, x=10, scale=_GLOBAL_CAPTION_SCALE)
    rows.append(gc)

    for task_name, sample_list in samples.items():
        # Spacer between task blocks
        rows.append(np.zeros((8, total_width, 3), dtype=np.uint8))

        # Task caption strip
        th = np.zeros((_TASK_HEADER_H, total_width, 3), dtype=np.uint8)
        _draw_left(th, f"task: {task_name}", x=10, scale=_TASK_HEADER_SCALE, thickness=1)
        rows.append(th)

        # Column header strip — labels what each column shows for THIS task
        ch = np.zeros((_COL_HEADER_H, total_width, 3), dtype=np.uint8)
        for idx, label in enumerate(_COLUMN_LABELS):
            _draw_centered(ch, label, idx * cell_size, (idx + 1) * cell_size,
                           scale=_COL_HEADER_SCALE)
        for idx in range(1, 4):
            cv2.line(ch, (idx * cell_size, 4), (idx * cell_size, _COL_HEADER_H - 4),
                     (90, 90, 90), 1, _LINE_TYPE)
        rows.append(ch)

        # Samples for this task
        for s in sample_list:
            tiles: list[np.ndarray] = []
            for key in ("clean", "degraded", "predicted"):
                img = _t_to_uint8_rgb(s[key])
                if img.shape[:2] != (cell_size, cell_size):
                    img = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
                tiles.append(img)
            delta = _delta_heatmap(s["predicted"], s["clean"])
            if delta.shape[:2] != (cell_size, cell_size):
                delta = cv2.resize(delta, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            rows.append(np.concatenate(tiles + [delta], axis=1))

    return np.concatenate(rows, axis=0)


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

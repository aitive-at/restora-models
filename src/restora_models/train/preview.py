"""Preview helpers: render multi-task grid + run temporal model inference.

The renderer is contract-stable across the per-frame-era / temporal-era
trainers — it consumes ``{task: [{clean, degraded, predicted}, ...]}``.

``make_temporal_preview_samples`` is the new bridge for the
``(B,7,3,H,W)`` model contract: it wraps each still ``clean`` frame
to a 7-frame window via ``replicate_to_window`` so the model sees its
expected input shape, then collects the central-frame outputs.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from restora_models.data.compound import AXES, DEGRADE_ORDER
from restora_models.data.degradations.registry import Degradation, build_degradation
from restora_models.data.window import replicate_to_window
from restora_models.data.video_window import VideoWindowDataset

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
      [global caption strip - step + timestamp]
      for each task:
        [blank spacer (8 px)]
        [task caption strip - full-width, names the task]
        [column header - Clean | Degraded | Predicted | |delta| heatmap]
        [task's samples - one per row, four columns each]

    Repeating the column header per task means every row of images is
    fully self-described (task + per-column meaning).

    Each cell is `cell_size x cell_size`; the grid is 4 * cell_size wide.
    """
    total_width = 4 * cell_size
    rows: list[np.ndarray] = []

    gc = np.zeros((_GLOBAL_CAPTION_H, total_width, 3), dtype=np.uint8)
    _draw_left(gc, caption, x=10, scale=_GLOBAL_CAPTION_SCALE)
    rows.append(gc)

    for task_name, sample_list in samples.items():
        rows.append(np.zeros((8, total_width, 3), dtype=np.uint8))

        th = np.zeros((_TASK_HEADER_H, total_width, 3), dtype=np.uint8)
        _draw_left(th, f"task: {task_name}", x=10, scale=_TASK_HEADER_SCALE, thickness=1)
        rows.append(th)

        ch = np.zeros((_COL_HEADER_H, total_width, 3), dtype=np.uint8)
        for idx, label in enumerate(_COLUMN_LABELS):
            _draw_centered(ch, label, idx * cell_size, (idx + 1) * cell_size,
                           scale=_COL_HEADER_SCALE)
        for idx in range(1, 4):
            cv2.line(ch, (idx * cell_size, 4), (idx * cell_size, _COL_HEADER_H - 4),
                     (90, 90, 90), 1, _LINE_TYPE)
        rows.append(ch)

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


# ---------------------------------------------------------------------
# Temporal-model inference helpers
# ---------------------------------------------------------------------

# The set of (label, config_vec, options) rows shown in the preview grid.
# Each row demonstrates one axis (or all) on the same set of clean frames.
_PREVIEW_TASKS: list[tuple[str, list[int], dict]] = [
    ("identity",      [0, 0, 0, 0, 0], {}),
    ("colorize-only", [1, 0, 0, 0, 0], {}),
    ("denoise-only",  [0, 1, 0, 0, 0], {}),
    ("sharpen-2x",    [0, 0, 1, 0, 0], {"sharpen_factor": 2}),
    ("sharpen-4x",    [0, 0, 1, 0, 0], {"sharpen_factor": 4}),
    ("dejpeg-only",   [0, 0, 0, 1, 0], {}),
    ("deblur-only",   [0, 0, 0, 0, 1], {}),
    ("all-on",        [1, 1, 1, 1, 1], {}),
]


def _build_axis_degradations() -> dict[str, Degradation]:
    return {
        "colorize": build_degradation("colorize", {}),
        "denoise":  build_degradation("denoise", {"sigma_range": [0.005, 0.05]}),
        "sharpen":  build_degradation("sharpen", {"factor_choices": [2, 4, 8]}),
        "dejpeg":   build_degradation("jpeg", {"quality_range": [20, 70]}),
        "deblur":   build_degradation("deblur", {"sigma_range": [1.0, 3.0]}),
    }


@torch.inference_mode()
def make_temporal_preview_samples(
    *,
    model: nn.Module,
    dataset: VideoWindowDataset,
    device: torch.device,
    sample_indices: list[int],
    seed: int = 0,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    """Run the temporal model on a small set of stills + render-ready samples.

    For each ``(label, config_vec, opts)`` row in ``_PREVIEW_TASKS`` and
    each ``sample_indices`` index, the central frame of the dataset's
    7-frame clip is used as the clean still. Per-axis degradations are
    applied in DEGRADE_ORDER, then the degraded single frame is replicated
    to a (1,7,3,H,W) window so the temporal model sees its expected shape.
    """
    out: dict[str, list[dict[str, torch.Tensor]]] = {label: [] for label, _, _ in _PREVIEW_TASKS}
    axis_degs = _build_axis_degradations()

    for label, vec, opts in _PREVIEW_TASKS:
        flags = dict(zip(AXES, vec))
        sharpen_override = None
        if "sharpen_factor" in opts:
            sharpen_override = build_degradation(
                "sharpen", {"factor_choices": [int(opts["sharpen_factor"])]})

        for i in sample_indices:
            sample = dataset[i % len(dataset)]
            clean_clip = sample["frames"]   # (7, 3, H, W)
            t_mid = clean_clip.shape[0] // 2
            clean_t = clean_clip[t_mid].contiguous()
            rng = random.Random((seed * 1_000_003) ^ i)
            rgb_np = clean_t.permute(1, 2, 0).contiguous().numpy().copy()
            for axis in DEGRADE_ORDER:
                if not flags[axis]:
                    continue
                deg = (sharpen_override
                       if (axis == "sharpen" and sharpen_override is not None)
                       else axis_degs[axis])
                rgb_np = deg.degrade(rgb_np, rng)
            degraded_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).contiguous()
            # Wrap single still -> (1, 7, 3, H, W) so the temporal model
            # sees its required 7-frame window.
            window = replicate_to_window(degraded_t).unsqueeze(0).to(device)
            cfg_t = torch.tensor([vec], dtype=torch.float32, device=device)
            pred = model(window, cfg_t)
            out[label].append({
                "clean": clean_t,
                "degraded": degraded_t,
                "predicted": pred.clamp(0, 1).squeeze(0).cpu(),
            })
    return out

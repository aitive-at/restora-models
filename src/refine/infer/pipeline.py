"""Inference pipeline: RGB-in, RGB-out, task-aware."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def pad_to_multiple(rgb: np.ndarray, *, multiple: int = 16,
                     mode: str = "reflect") -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    padded = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode)
    return padded, (pad_t, pad_b, pad_l, pad_r)


def unpad(img: np.ndarray, pad_t: int, pad_b: int, pad_l: int, pad_r: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[pad_t : h - pad_b, pad_l : w - pad_r]


class MultiTaskRefinerPipeline:
    def __init__(self, model: nn.Module, *, task_name_to_id: dict[str, int],
                 device: torch.device | None = None) -> None:
        self.task_name_to_id = dict(task_name_to_id)
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.train(False)

    @torch.inference_mode()
    def process(self, img_bgr: np.ndarray, *, task: str) -> np.ndarray:
        if task not in self.task_name_to_id:
            raise ValueError(f"unknown task {task!r}; have {list(self.task_name_to_id)}")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_padded, pads = pad_to_multiple(rgb, multiple=16, mode="reflect")
        t = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        task_id = torch.tensor([self.task_name_to_id[task]], dtype=torch.long, device=self.device)
        out = self.model(t, task_id).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = unpad(out, *pads)
        return (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255.0).round().clip(0, 255).astype(np.uint8)


def load_pipeline(checkpoint: str | Path, *, device: torch.device | None = None) -> MultiTaskRefinerPipeline:
    from refine.config import ModelConfig
    from refine.models import build_model

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg_dict = cfg_dict.get("model") or {}
    mcfg = ModelConfig(**mcfg_dict)
    task_map = payload.get("task_map") or {}
    task_name_to_id = task_map.get("tasks") or {"colorize": 0}
    model = build_model(mcfg, num_tasks=len(task_name_to_id))
    model.load_state_dict(payload["model"])
    return MultiTaskRefinerPipeline(model, task_name_to_id=task_name_to_id, device=device)

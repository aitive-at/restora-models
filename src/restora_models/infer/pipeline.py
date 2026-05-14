"""Inference pipeline: RGB-in, RGB-out, conditioning-vector-aware."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from restora_models.data.compound import AXES


def pad_to_multiple(rgb, *, multiple: int = 16, mode: str = "reflect"):
    h, w = rgb.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    padded = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode)
    return padded, (pad_t, pad_b, pad_l, pad_r)


def unpad(img, pad_t, pad_b, pad_l, pad_r):
    h, w = img.shape[:2]
    return img[pad_t : h - pad_b, pad_l : w - pad_r]


def _config_to_tensor(config, device) -> torch.Tensor:
    """config can be: dict[str, bool], list[float]/list[int], or torch.Tensor."""
    if isinstance(config, torch.Tensor):
        vec = config.float().to(device)
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        return vec
    if isinstance(config, dict):
        vec = [float(bool(config.get(a, False))) for a in AXES]
    else:  # list / tuple
        vec = [float(v) for v in config]
        if len(vec) != len(AXES):
            raise ValueError(f"config must have {len(AXES)} values; got {len(vec)}")
    return torch.tensor([vec], dtype=torch.float32, device=device)


class CompoundRefinerPipeline:
    def __init__(self, model: nn.Module, *, device: torch.device | None = None) -> None:
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.train(False)

    @torch.inference_mode()
    def process(self, img_bgr: np.ndarray, *, config) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_padded, pads = pad_to_multiple(rgb, multiple=16, mode="reflect")
        t = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        cfg_t = _config_to_tensor(config, self.device)
        out = self.model(t, cfg_t).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = unpad(out, *pads)
        return (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255.0).round().clip(0, 255).astype(np.uint8)


def load_pipeline(checkpoint: str | Path, *, device: torch.device | None = None) -> CompoundRefinerPipeline:
    from restora_models.config import ModelConfig
    from restora_models.models import build_model

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg = ModelConfig(**(cfg_dict.get("model") or {}))
    model = build_model(mcfg, num_axes=len(AXES))
    model.load_state_dict(payload["model"])
    return CompoundRefinerPipeline(model, device=device)

"""Inference pipeline: replicates the original ColorizationPipeline LAB routing."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ColorizationPipeline:
    def __init__(
        self, model: nn.Module, *, input_size: int, device: torch.device | None = None
    ) -> None:
        self.input_size = int(input_size)
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.train(False)

    @torch.inference_mode()
    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None:
            raise ValueError("img is None")
        h, w = img_bgr.shape[:2]
        img = img_bgr.astype(np.float32) / 255.0
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        l_low = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        gray_lab = np.concatenate([l_low, np.zeros_like(l_low), np.zeros_like(l_low)], axis=-1)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)

        tensor = (
            torch.from_numpy(gray_rgb.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        output_ab = self.model(tensor).cpu()
        output_ab_resized = (
            F.interpolate(output_ab, size=(h, w), mode="bilinear", align_corners=False)[0]
            .float()
            .numpy()
            .transpose(1, 2, 0)
        )
        output_lab = np.concatenate([orig_l, output_ab_resized], axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        return (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)


def load_pipeline(
    checkpoint: str | Path,
    *,
    input_size: int,
    device: torch.device | None = None,
) -> ColorizationPipeline:
    from coliraz.config import ModelConfig
    from coliraz.models import build_ddcolor

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg_dict = cfg_dict.get("model") or {"input_size": input_size}
    mcfg = ModelConfig(**mcfg_dict)
    model = build_ddcolor(mcfg, pretrained=False)
    model.load_state_dict(payload["model"])
    return ColorizationPipeline(model, input_size=input_size, device=device)

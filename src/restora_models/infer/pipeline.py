"""Inference pipeline for the temporal restoration model.

Handles both single-image inference (replicates the frame 7x) and
directory-of-frames inference (sliding 7-frame window with edge-replicate
at boundaries).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from restora_models.data.compound import AXES
from restora_models.data.window import replicate_to_window


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _to_5axis(config: dict[str, bool]) -> torch.Tensor:
    """Map a {'colorize': True, ...} dict to a (5,) float vector matching AXES order."""
    vec = torch.zeros(len(AXES), dtype=torch.float32)
    for i, ax in enumerate(AXES):
        if config.get(ax, False):
            vec[i] = 1.0
    return vec


class VideoPipeline:
    """Wrap a TemporalRestora checkpoint for end-user inference.

    Inputs are BGR uint8 numpy arrays (OpenCV convention). Outputs are
    BGR uint8. Color conversion to/from the model's RGB float [0,1] domain
    is handled internally.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cuda"):
        model.train(False)
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = self.model.to(self.device)

    @classmethod
    def from_checkpoint(cls, ckpt_path: Path, device: str | torch.device = "cuda") -> "VideoPipeline":
        from restora_models.config import ModelConfig
        from restora_models.models.registry import build_model
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg_dict = (payload.get("extra") or {}).get("cfg", {})
        mtype = (cfg_dict.get("model") or {}).get("type", "temporal_restora_small")
        m = build_model(ModelConfig(type=mtype), num_axes=len(AXES))
        m.load_state_dict(payload["model"])
        return cls(model=m, device=device)

    @torch.inference_mode()
    def process_image(self, image_bgr: np.ndarray, *, config: dict[str, bool]) -> np.ndarray:
        """Single still: replicate to a 7-frame clip and run the model."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) BGR uint8, got {image_bgr.shape}")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        # Pad to multiple of 16 (network requires divisible-by-16 spatial)
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        clip = replicate_to_window(img_t, num_frames=7).unsqueeze(0).to(self.device)
        cfg_vec = _to_5axis(config).unsqueeze(0).to(self.device)
        out = self.model(clip, cfg_vec).clamp(0, 1).squeeze(0)
        out = out.permute(1, 2, 0).cpu().numpy()
        if ph or pw:
            out = out[:h, :w]
        return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    @torch.inference_mode()
    def process_directory(
        self, in_dir: Path, out_dir: Path, *, config: dict[str, bool],
    ) -> None:
        """Sliding 7-frame window over frame_*.png in alphabetical order."""
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_files = sorted(p for p in in_dir.iterdir()
                              if p.is_file() and p.suffix.lower() in _IMG_EXTS)
        if not frame_files:
            raise ValueError(f"no image files found in {in_dir}")
        frames_bgr: list[np.ndarray] = []
        for p in frame_files:
            arr = cv2.imread(str(p))
            if arr is None:
                raise RuntimeError(f"failed to read {p}")
            frames_bgr.append(arr)

        cfg_vec = _to_5axis(config).unsqueeze(0).to(self.device)
        center_index = 3
        n = len(frames_bgr)
        h, w = frames_bgr[0].shape[:2]
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16

        for i in range(n):
            window: list[np.ndarray] = []
            for k in range(7):
                src_idx = max(0, min(n - 1, i - center_index + k))
                fr = frames_bgr[src_idx]
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                if ph or pw:
                    rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
                window.append(rgb)
            clip = torch.from_numpy(np.stack(window)).permute(0, 3, 1, 2).contiguous()
            clip = clip.unsqueeze(0).to(self.device)
            out = self.model(clip, cfg_vec).clamp(0, 1).squeeze(0)
            out = out.permute(1, 2, 0).cpu().numpy()
            if ph or pw:
                out = out[:h, :w]
            out_path = out_dir / frame_files[i].name
            cv2.imwrite(str(out_path), cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

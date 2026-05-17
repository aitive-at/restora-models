"""MPEG transcode degradation via ffmpeg subprocess.

For VHS/broadcast-era footage realism. Encodes the clip to a tempfile,
decodes back, returns. Per-clip (needs full GOP for inter-frame coding).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


class MpegTranscodeDegradation:
    def __init__(self, codec: str = "mpeg1video", bitrate_kbps: int = 300):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        self.codec = codec
        self.bitrate_kbps = bitrate_kbps

    def apply_clip(self, clip: torch.Tensor) -> torch.Tensor:
        import cv2
        if clip.dim() != 4 or clip.shape[1] != 3:
            raise ValueError(f"expected (T,3,H,W), got {tuple(clip.shape)}")
        t, _, h, w = clip.shape
        arr = (clip.permute(0, 2, 3, 1).clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as td:
            inp_path = Path(td) / "in.mp4"
            out_path = Path(td) / "out.mp4"
            writer = cv2.VideoWriter(str(inp_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
            for f in arr:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-i", str(inp_path),
                   "-c:v", self.codec, "-b:v", f"{self.bitrate_kbps}k",
                   str(out_path)]
            subprocess.run(cmd, check=True)
            cap = cv2.VideoCapture(str(out_path))
            decoded = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
        if len(decoded) == 0:
            return clip.clone()
        out_arr = np.stack(decoded[:t]).astype(np.float32) / 255.0
        if out_arr.shape[0] < t:
            pad = np.repeat(out_arr[-1:], t - out_arr.shape[0], axis=0)
            out_arr = np.concatenate([out_arr, pad], axis=0)
        return torch.from_numpy(out_arr).permute(0, 3, 1, 2).to(clip.device).to(clip.dtype)

"""Tests for MPEG/H.263 transcode degradation (ffmpeg subprocess)."""
import shutil

import pytest
import torch

from restora_models.data.degradations.mpeg_transcode import MpegTranscodeDegradation


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mpeg_transcode_clip_shape():
    deg = MpegTranscodeDegradation(codec="mpeg1video", bitrate_kbps=200)
    clip = torch.rand(7, 3, 64, 64)
    out = deg.apply_clip(clip)
    assert out.shape == clip.shape


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mpeg_transcode_introduces_artifacts():
    deg = MpegTranscodeDegradation(codec="mpeg1video", bitrate_kbps=150)
    clip = torch.rand(7, 3, 64, 64)
    out = deg.apply_clip(clip)
    diff = (out - clip).abs().mean().item()
    assert diff > 0.001, f"transcode too gentle: {diff}"

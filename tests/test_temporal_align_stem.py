"""Tests for TemporalAlignStem in models/temporal_align_stem.py."""
import torch

from restora_models.models.temporal_align_stem import TemporalAlignStem


def test_align_stem_output_shape():
    stem = TemporalAlignStem().eval()
    frames = torch.rand(2, 7, 3, 64, 64)
    out = stem(frames)
    assert out.shape == (2, 28, 64, 64), f"got {tuple(out.shape)}"


def test_align_stem_identical_frames_path():
    stem = TemporalAlignStem().eval()
    img = torch.rand(1, 3, 64, 64)
    frames = img.unsqueeze(1).expand(1, 7, 3, 64, 64).contiguous()
    out = stem(frames)
    rgb_part = out[:, :21].view(1, 7, 3, 64, 64)
    for k in range(7):
        assert rgb_part[:, k].shape == img.shape
    mask_part = out[:, 21:]
    assert mask_part.shape == (1, 7, 64, 64)

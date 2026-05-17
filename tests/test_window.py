"""Tests for replicate_to_window (single image -> 7-frame clip)."""
import torch

from restora_models.data.window import replicate_to_window


def test_replicate_single_image_to_7_frame_clip():
    img = torch.rand(3, 32, 32)
    clip = replicate_to_window(img, num_frames=7)
    assert clip.shape == (7, 3, 32, 32)
    for k in range(7):
        assert torch.equal(clip[k], img)


def test_replicate_short_clip_pads_edges():
    short = torch.rand(3, 3, 16, 16)
    clip = replicate_to_window(short, num_frames=7, center_index=3)
    assert clip.shape == (7, 3, 16, 16)
    assert torch.equal(clip[3], short[1])

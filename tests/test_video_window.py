"""Tests for VideoSubDataset protocol + VideoWindowDataset facade."""
import torch
from torch.utils.data import Dataset

from restora_models.data.video_window import VideoSubDataset, VideoWindowDataset


class _FakeSub(Dataset):
    name = "fake"

    def __init__(self, n: int, seed: int):
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(self.seed + idx)
        return {
            "frames": torch.rand(7, 3, 32, 32, generator=g),
            "source": self.name,
            "key": f"{self.name}_{idx}",
        }


def test_video_window_concat_lengths():
    a = _FakeSub(n=10, seed=0)
    b = _FakeSub(n=20, seed=1)
    ds = VideoWindowDataset(sub_datasets=[a, b], weights=[1.0, 1.0])
    assert len(ds) == 30


def test_video_window_returns_canonical_shape():
    a = _FakeSub(n=5, seed=0)
    ds = VideoWindowDataset(sub_datasets=[a], weights=[1.0])
    sample = ds[0]
    assert sample["frames"].shape == (7, 3, 32, 32)
    assert sample["frames"].dtype == torch.float32
    assert "source" in sample
    assert "key" in sample


def test_video_window_sample_random_returns_dict():
    """sample_random returns the dict from a random sub-dataset."""
    a = _FakeSub(n=10, seed=0)
    b = _FakeSub(n=10, seed=1)
    ds = VideoWindowDataset(sub_datasets=[a, b], weights=[1.0, 1.0])
    for _ in range(20):
        s = ds.sample_random()
        assert s["frames"].shape == (7, 3, 32, 32)
        assert s["source"] == "fake"

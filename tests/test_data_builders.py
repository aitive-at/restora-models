"""Tests for data builder factory."""
from pathlib import Path

import pytest

from restora_models.data.builders import build_video_window_dataset


def test_build_unknown_type_raises():
    with pytest.raises(KeyError):
        build_video_window_dataset([{"type": "no_such_type", "weight": 1.0}])


def test_build_reds_dataset(tmp_path: Path):
    (tmp_path / "train_sharp" / "000").mkdir(parents=True)
    sources = [{"type": "reds", "root": str(tmp_path), "split": "train_sharp",
                "window": 7, "stride": 1, "crop": 32, "weight": 1.0}]
    ds = build_video_window_dataset(sources)
    assert len(ds) == 0  # no frames -> 0 windows

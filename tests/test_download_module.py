"""Smoke tests for restora_models.data.download — module wiring, no network calls."""
from __future__ import annotations

import pytest

from restora_models.data.download import (
    NUM_PARQUET_SHARDS, _DATASETS, _parquet_filename, list_datasets,
)


def test_list_datasets_known_subsets():
    names = list_datasets()
    assert set(names) == {
        "relaion2B-multi-aesthetic",
        "laion2B-en-aesthetic",
        "relaion1B-nolang-aesthetic",
    }


def test_num_shards_constant():
    """The LAION-aesthetic subsets are always sharded into exactly 128 parts."""
    assert NUM_PARQUET_SHARDS == 128


@pytest.mark.parametrize("dataset", list(_DATASETS))
def test_parquet_filename_format(dataset):
    """Filenames follow part-NNNNN-{uuid}-c000.snappy.parquet."""
    ds = _DATASETS[dataset]
    name = _parquet_filename(ds, 0)
    assert name == f"part-00000-{ds.uuid}-c000.snappy.parquet"
    name127 = _parquet_filename(ds, 127)
    assert name127 == f"part-00127-{ds.uuid}-c000.snappy.parquet"


def test_dataset_uuids_are_distinct():
    """Each subset must have a unique UUID — otherwise filenames collide."""
    uuids = {ds.uuid for ds in _DATASETS.values()}
    assert len(uuids) == len(_DATASETS)


def test_download_function_rejects_unknown_dataset():
    from restora_models.data.download import download_laion_aesthetic
    with pytest.raises(ValueError, match="unknown dataset"):
        download_laion_aesthetic("not-a-real-dataset", output_dir="/tmp/nowhere")

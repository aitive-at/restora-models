from pathlib import Path

import pytest

from coliraz.config import (
    Config,
    deep_merge,
    expand_loss_preset,
    load_config,
)


def test_expand_loss_preset_standard():
    losses = expand_loss_preset("standard")
    names = [l.name for l in losses]
    assert "l1_ab" in names
    assert "perceptual_vgg16bn" in names
    assert "colorfulness" in names


def test_expand_loss_preset_minimal():
    losses = expand_loss_preset("minimal")
    assert [l.name for l in losses] == ["l1_ab"]


def test_deep_merge_overrides_leaf_keys():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    over = {"a": {"b": 99}}
    out = deep_merge(base, over)
    assert out == {"a": {"b": 99, "c": 2}, "d": 3}


def test_load_config_with_preset(tmp_path: Path):
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text(
        """
data: { root: /tmp/x }
losses: !preset minimal
"""
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert [l.name for l in cfg.losses] == ["l1_ab"]
    assert cfg.data.root == "/tmp/x"


def test_load_config_chained_defaults(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("data: { root: /a, val_fraction: 0.05 }\nlosses: !preset minimal\n")
    (tmp_path / "child.yaml").write_text(
        "defaults: base.yaml\ndata: { val_fraction: 0.01 }\n"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.data.root == "/a"
    assert cfg.data.val_fraction == 0.01


def test_cli_overrides_win(tmp_path: Path):
    (tmp_path / "x.yaml").write_text("data: { root: /a }\nlosses: !preset minimal\n")
    cfg = load_config(tmp_path / "x.yaml", overrides={"data": {"root": "/b"}})
    assert cfg.data.root == "/b"


def test_required_fields_raise():
    with pytest.raises(Exception):
        Config.model_validate({})

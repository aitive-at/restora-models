from pathlib import Path

from typer.testing import CliRunner

from coliraz.cli import app
from coliraz.config import load_config

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_help_top_level():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    out = r.stdout
    assert "train" in out and "infer" in out and "export" in out and "scan-data" in out


def test_help_train():
    r = runner.invoke(app, ["train", "--help"])
    assert r.exit_code == 0


def test_scan_data_writes_manifest(tmp_image_dir):
    r = runner.invoke(app, ["scan-data", "--root", str(tmp_image_dir)])
    assert r.exit_code == 0
    assert (tmp_image_dir / ".coliraz-manifest.txt").exists()


def test_tiny_yaml_loads():
    cfg = load_config(ROOT / "tiny.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "tiny"
    assert cfg.data.root == "/tmp"


def test_large_yaml_loads():
    cfg = load_config(ROOT / "large.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "large"

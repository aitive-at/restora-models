from pathlib import Path

from typer.testing import CliRunner

from refine.cli import app

runner = CliRunner()


def test_help_top_level():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for sub in ("train", "infer", "export", "scan-data", "info"):
        assert sub in r.stdout


def test_help_top_level_includes_info():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "info" in r.stdout
    assert "list-tasks" not in r.stdout


def test_help_train():
    r = runner.invoke(app, ["train", "--help"])
    assert r.exit_code == 0


def test_scan_data_writes_manifest(tmp_image_dir):
    r = runner.invoke(app, ["scan-data", "--root", str(tmp_image_dir)])
    assert r.exit_code == 0
    assert (tmp_image_dir / ".refine-manifest.txt").exists()


def test_help_infer_has_axis_flags():
    r = runner.invoke(app, ["infer", "--help"])
    assert r.exit_code == 0
    assert "--color" in r.stdout
    assert "--denoise" in r.stdout
    assert "--sharp" in r.stdout
    assert "--dejpeg" in r.stdout
    assert "--deblur" in r.stdout

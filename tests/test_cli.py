from pathlib import Path

from typer.testing import CliRunner

from refine.cli import app

runner = CliRunner()


def test_help_top_level():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for sub in ("train", "infer", "export", "scan-data", "list-tasks"):
        assert sub in r.stdout


def test_help_train():
    r = runner.invoke(app, ["train", "--help"])
    assert r.exit_code == 0


def test_scan_data_writes_manifest(tmp_image_dir):
    r = runner.invoke(app, ["scan-data", "--root", str(tmp_image_dir)])
    assert r.exit_code == 0
    assert (tmp_image_dir / ".refine-manifest.txt").exists()


def test_list_tasks(tmp_path):
    ROOT = Path(__file__).resolve().parents[1] / "configs"
    r = runner.invoke(app, ["list-tasks", "--config", str(ROOT / "tiny.yaml"),
                            "--data", str(tmp_path)])
    assert r.exit_code == 0
    assert "colorize" in r.stdout
    assert "sr_x4" in r.stdout

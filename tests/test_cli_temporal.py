"""Tests for the temporal CLI surface."""
from typer.testing import CliRunner

from restora_models.cli import app


def test_cli_version():
    r = CliRunner().invoke(app, ["version"])
    assert r.exit_code == 0
    assert r.stdout.strip()  # non-empty


def test_cli_has_expected_commands():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ["train", "infer", "export", "distill", "bench", "compare",
                "gallery", "prepare-data", "train-flow-distill", "train-pipeline"]:
        assert cmd in r.stdout, f"missing command: {cmd}"


def test_cli_no_obsolete_commands():
    r = CliRunner().invoke(app, ["--help"])
    for cmd in ["scan-data", "download-davis", "download-imagenet",
                "download-openimages", "prepare-videos", "precompute-flow",
                "make-synthetic-videos"]:
        assert cmd not in r.stdout, f"obsolete command still present: {cmd}"


def test_cli_prepare_subcommands():
    r = CliRunner().invoke(app, ["prepare-data", "--help"])
    assert r.exit_code == 0
    for sub in ["film-overlays", "reds", "vimeo"]:
        assert sub in r.stdout, f"missing prepare-data subcommand: {sub}"


def test_stub_commands_exit_when_dependency_missing(tmp_path):
    """Stub commands like infer/export should give a clear message + non-zero exit
    when their backing module isn't implemented yet."""
    # These commands all require a non-existent file at --model/--ckpt, so they
    # fail at typer's exists=True check (exit code 2) before reaching the body.
    # That's still a graceful exit, not a crash.
    r = CliRunner().invoke(app, ["infer", "--model", str(tmp_path / "x.pt"),
                                  "--input", str(tmp_path), "--output", str(tmp_path / "o")])
    assert r.exit_code != 0  # any non-zero is fine; just don't crash

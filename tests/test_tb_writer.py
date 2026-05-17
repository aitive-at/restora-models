"""Tests for ``restora_models.train.tb.TensorBoardWriter``.

Covers happy-path construction + event-file emission, scalar/image
logging round-trip, and the graceful-degradation path when the
``SummaryWriter`` import (or constructor) fails — every public method
must become a no-op without raising.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


def _list_event_files(tb_dir: Path) -> list[Path]:
    """TensorBoard event files start with ``events.out.tfevents.``."""
    return sorted(tb_dir.glob("events.out.tfevents.*"))


def test_construct_and_close_creates_event_file(tmp_path: Path) -> None:
    from restora_models.train.tb import TensorBoardWriter

    tb_dir = tmp_path / "tb"
    assert not tb_dir.exists()

    writer = TensorBoardWriter(tmp_path)
    assert tb_dir.is_dir()
    writer.close()

    events = _list_event_files(tb_dir)
    assert events, f"expected at least one event file in {tb_dir}"


def test_log_scalars_and_image_round_trip(tmp_path: Path) -> None:
    from restora_models.train.tb import TensorBoardWriter

    with TensorBoardWriter(tmp_path, flush_every_s=0.0) as writer:
        writer.log_scalars(
            step=1,
            scalars={
                "loss/total": 0.42,
                "loss/l1_rgb": 0.18,
                "metric/psnr/colorize": 17.5,
                "train/lr": 1e-3,
                "train/img_per_s": 12.7,
                # Should be silently skipped (non-float / non-finite):
                "loss/weights": [0.1, 0.2],
                "loss/nan": float("nan"),
                "loss/inf": float("inf"),
            },
        )
        img = np.random.default_rng(0).integers(
            0, 256, size=(64, 96, 3), dtype=np.uint8)
        writer.log_image(step=1, tag="preview/grid", image_hwc_uint8=img)
        writer.log_scalars(step=2, scalars={"loss/total": 0.30})
        writer.flush()

    events = _list_event_files(tmp_path / "tb")
    assert events, "expected event file after logging"
    # Non-empty: TensorBoard always writes a file header even with no
    # events, so the byte-count check guards against the writer falling
    # back to no-op silently.
    total_bytes = sum(p.stat().st_size for p in events)
    assert total_bytes > 256, (
        f"event file looks empty/header-only ({total_bytes} bytes)")


def test_bad_image_shape_does_not_raise(tmp_path: Path) -> None:
    from restora_models.train.tb import TensorBoardWriter

    with TensorBoardWriter(tmp_path) as writer:
        # Wrong rank — must be tolerated (logged at debug, no raise).
        writer.log_image(step=0, tag="bad/rank",
                         image_hwc_uint8=np.zeros((32, 32), dtype=np.uint8))
        # Wrong channel count.
        writer.log_image(step=0, tag="bad/chan",
                         image_hwc_uint8=np.zeros((32, 32, 4), dtype=np.uint8))
        # Wrong type.
        writer.log_image(step=0, tag="bad/type",
                         image_hwc_uint8="not an array")  # type: ignore[arg-type]


def test_disabled_when_summarywriter_import_missing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patching the module-level ``_SummaryWriter`` to ``None`` should
    flip the writer into the disabled / no-op path. Every public method
    must complete silently and no ``tb/`` directory should be written."""
    import restora_models.train.tb as tb_mod

    # Force a fresh import in case other tests already touched the
    # module — we need the patch to land on the same binding the class
    # body reads at construction time.
    importlib.reload(tb_mod)
    monkeypatch.setattr(tb_mod, "_SummaryWriter", None)
    monkeypatch.setattr(tb_mod, "_TB_IMPORT_ERROR",
                        ImportError("simulated missing tensorboard"))

    writer = tb_mod.TensorBoardWriter(tmp_path)
    assert writer._disabled is True
    # No tb dir is created on the disabled path.
    assert not (tmp_path / "tb").exists()

    # All public methods must be silent no-ops.
    writer.log_scalars(step=0, scalars={"loss/total": 1.0})
    writer.log_image(
        step=0, tag="preview/grid",
        image_hwc_uint8=np.zeros((8, 8, 3), dtype=np.uint8))
    writer.flush()
    writer.close()

    # Reload once more so subsequent tests get the un-patched module.
    importlib.reload(tb_mod)


def test_disabled_when_summarywriter_constructor_raises(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``_SummaryWriter(...)`` raises at construction, the writer
    must catch + log + fall back to no-op (defensive against e.g.
    filesystem permission errors on the target dir)."""
    import restora_models.train.tb as tb_mod

    importlib.reload(tb_mod)

    class _BoomWriter:
        def __init__(self, *_a, **_kw):
            raise RuntimeError("simulated SummaryWriter failure")

    monkeypatch.setattr(tb_mod, "_SummaryWriter", _BoomWriter)

    writer = tb_mod.TensorBoardWriter(tmp_path)
    assert writer._disabled is True
    # The tb dir is created before the SummaryWriter call — that's fine.
    writer.log_scalars(step=0, scalars={"loss/total": 1.0})
    writer.close()

    importlib.reload(tb_mod)

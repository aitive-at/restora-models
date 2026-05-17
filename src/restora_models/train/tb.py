"""TensorBoard writer — additive logging that mirrors the rich-console UI.

The trainer owns one ``TensorBoardWriter`` per run, instantiated once
``out_dir`` is known. The writer creates ``<out_dir>/tb/`` and streams
scalars on every tick + preview images on every preview emit. It is
deliberately *dumb*: tag namespacing (``loss/...``, ``metric/psnr/...``,
``train/...``) is the trainer's job — the writer just forwards whatever
flat ``{tag: float}`` dict it gets. That keeps it reusable for future
sub-trainers (distillation, eval-only) without forking conventions.

Failure mode: if ``torch.utils.tensorboard`` (or the underlying
``tensorboard`` wheel) can't import, every public method becomes a
silent no-op and a single warning is logged at construction. The
trainer never needs to know — the additive logging just disappears.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy/guarded import. We want construction of `TensorBoardWriter` to
# succeed even when the tensorboard wheel is absent or broken — the
# trainer treats this writer as best-effort instrumentation.
try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _TB_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - exercised via patching
    _SummaryWriter = None  # type: ignore[assignment]
    _TB_IMPORT_ERROR = _exc


class TensorBoardWriter:
    """Per-run TensorBoard sink. See module docstring for tag conventions.

    Construction creates ``<run_dir>/tb/`` and opens a ``SummaryWriter``
    pointed at it. Auto-flush every ``flush_every_s`` wall-clock seconds
    keeps the dashboard usable during long runs without per-tick fsync.
    """

    def __init__(self, run_dir: str | Path, *, flush_every_s: float = 30.0) -> None:
        self._disabled = False
        self._writer: Any = None
        self._flush_every_s = float(flush_every_s)
        self._last_flush_t = time.perf_counter()

        if _SummaryWriter is None:
            self._disabled = True
            logger.warning(
                "TensorBoardWriter disabled: failed to import "
                "torch.utils.tensorboard (%s). Scalar/image logging will "
                "be a no-op for this run.", _TB_IMPORT_ERROR,
            )
            return

        try:
            self.tb_dir = Path(run_dir) / "tb"
            self.tb_dir.mkdir(parents=True, exist_ok=True)
            self._writer = _SummaryWriter(log_dir=str(self.tb_dir))
        except Exception as exc:  # pragma: no cover - defensive
            self._disabled = True
            self._writer = None
            logger.warning(
                "TensorBoardWriter disabled: SummaryWriter init failed (%s). "
                "Scalar/image logging will be a no-op for this run.", exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        """Forward a *pre-tagged* flat dict (e.g. ``{"loss/total": 0.24,
        "metric/psnr/colorize": 17.5}``) to the underlying writer.

        Non-float values are silently skipped so the trainer can pass its
        raw log dict without filtering. ``NaN`` / ``inf`` are also
        skipped — TensorBoard tolerates them but they make autoscaled
        plots unreadable.
        """
        if self._disabled or self._writer is None:
            return
        for tag, value in scalars.items():
            if not isinstance(value, (int, float)):
                continue
            v = float(value)
            if not np.isfinite(v):
                continue
            try:
                self._writer.add_scalar(tag, v, global_step=int(step))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("tb add_scalar(%s) failed: %s", tag, exc)
        self._maybe_flush()

    def log_image(self, step: int, tag: str, image_hwc_uint8: np.ndarray) -> None:
        """Log an ``(H, W, 3) uint8`` RGB image (the format returned by
        ``render_multitask_grid``). ``dataformats="HWC"`` lets the
        SummaryWriter consume it without an extra transpose copy."""
        if self._disabled or self._writer is None:
            return
        if not isinstance(image_hwc_uint8, np.ndarray):
            logger.debug("tb log_image(%s): expected ndarray, got %s",
                         tag, type(image_hwc_uint8))
            return
        if image_hwc_uint8.ndim != 3 or image_hwc_uint8.shape[2] != 3:
            logger.debug("tb log_image(%s): expected (H,W,3), got shape %s",
                         tag, image_hwc_uint8.shape)
            return
        try:
            self._writer.add_image(
                tag, image_hwc_uint8,
                global_step=int(step), dataformats="HWC",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("tb add_image(%s) failed: %s", tag, exc)
        self._maybe_flush()

    def flush(self) -> None:
        if self._disabled or self._writer is None:
            return
        try:
            self._writer.flush()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("tb flush failed: %s", exc)
        self._last_flush_t = time.perf_counter()

    def close(self) -> None:
        if self._disabled or self._writer is None:
            return
        try:
            self._writer.flush()
            self._writer.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("tb close failed: %s", exc)
        finally:
            self._writer = None

    # ------------------------------------------------------------------
    # Context manager — `with TensorBoardWriter(run_dir) as tb:` closes
    # cleanly on exception so the final event file is always flushed.
    # ------------------------------------------------------------------

    def __enter__(self) -> "TensorBoardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_flush(self) -> None:
        if self._flush_every_s <= 0:
            return
        now = time.perf_counter()
        if (now - self._last_flush_t) >= self._flush_every_s:
            self.flush()


__all__ = ["TensorBoardWriter"]

"""Per-sample compound video-clip degradation wrapper.

This wrapper lifts the body of `Trainer._degrade_batch` (which currently
runs serially on the trainer's main process between forward passes) into
the DataLoader worker side so the ~250 numpy/opencv ops per step can be
parallelised across ``num_workers`` worker processes.

Design notes
------------
- Wraps a :class:`VideoWindowDataset` (or any object yielding
  ``{"frames": (T,3,H,W) float tensor, ...}``).
- ``__getitem__`` runs the *exact same pipeline* the trainer used:
  per-frame axis degradations (deblur/denoise/sharpen/dejpeg/colorize),
  optional film overlay, optional film color cast, optional per-clip
  gate weave, optional per-clip mpeg transcode.
- Per-worker RNG: each call seeds a fresh ``random.Random`` keyed by
  ``(base_seed, worker_id, idx, draw_counter)``. We use a single counter
  inside the call so consecutive degradations see independent draws.

Behaviour vs trainer
--------------------
The legacy `Trainer._degrade_batch` shared **one** `random.Random` across
*all* samples and batches. After this refactor each sample gets its own
deterministic seed derived from (worker_id, base_seed, idx). The
**distribution** of axis activations and degradation parameters is
preserved (same ``_sample_axes`` logic, same probabilities, same
``DEGRADE_ORDER``), but bit-equivalence across the refactor boundary is
impossible because workers now sample independently. That is acceptable
- the trainer was producing essentially-random batches before too.
"""
from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

_logger = logging.getLogger(__name__)
from typing import Any

import torch
from torch.utils.data import Dataset, get_worker_info

from .compound import AXES, DEGRADE_ORDER
from .degradations import colorization as _colorization  # noqa: F401
from .degradations import deblur as _deblur  # noqa: F401
from .degradations import denoise as _denoise  # noqa: F401
from .degradations import jpeg as _jpeg  # noqa: F401
from .degradations import superres as _superres  # noqa: F401
from .degradations.film_color_cast import FilmColorCastDegradation
from .degradations.film_overlay import FilmOverlayDegradation
from .degradations.gate_weave import GateWeaveDegradation
from .degradations.mpeg_transcode import MpegTranscodeDegradation
from .degradations.registry import Degradation, build_degradation

# Map axis -> registry name (mirrors `_AXIS_TO_REG` in trainer.py).
_AXIS_TO_REG = {
    "colorize": "colorize",
    "denoise":  "denoise",
    "sharpen":  "sharpen",
    "dejpeg":   "jpeg",
    "deblur":   "deblur",
}


# ---------------------------------------------------------------------------
# helpers (mirror trainer.py)
# ---------------------------------------------------------------------------

def _build_per_frame_degradations() -> dict[str, Degradation]:
    """Instantiate one Degradation per axis (matches `_build_per_frame_degradations` in trainer.py)."""
    deg_cfg = {
        "colorize": {},
        "denoise":  {"sigma_range": [0.005, 0.05]},
        "sharpen":  {"factor_choices": [2, 4, 8]},
        "dejpeg":   {"quality_range": [20, 70]},
        "deblur":   {"sigma_range": [1.0, 3.0], "motion_prob": 0.2},
    }
    return {axis: build_degradation(_AXIS_TO_REG[axis], deg_cfg[axis]) for axis in AXES}


def _sample_axes(rng: random.Random, identity_prob: float = 0.15) -> set[str]:
    """Sample task set with balanced single/compound/identity distribution.

    Mirrors `Trainer._sample_axes`:
      - identity_prob (default 0.15): empty set
      - 35% single random axis
      - 35% two axes
      - 15% three+ axes (3..len(AXES))
    """
    r = rng.random()
    if r < identity_prob:
        return set()
    remaining = 1.0 - identity_prob
    p_single = 0.35 / remaining
    p_two = 0.35 / remaining
    r2 = rng.random()
    if r2 < p_single:
        n = 1
    elif r2 < p_single + p_two:
        n = 2
    else:
        n = rng.randint(3, len(AXES))
    return set(rng.sample(list(AXES), n))


def _make_config_vec(active: set[str]) -> torch.Tensor:
    vec = torch.zeros(len(AXES))
    for i, ax in enumerate(AXES):
        if ax in active:
            vec[i] = 1.0
    return vec


def _apply_per_frame_degradations(
    clip: torch.Tensor,
    active_axes: set[str],
    per_frame_degs: dict[str, Degradation],
    rng: random.Random,
) -> torch.Tensor:
    """Apply each active axis to every frame (matches trainer._apply_per_frame_degradations)."""
    if not active_axes:
        return clip.clone()
    out = []
    for k in range(clip.shape[0]):
        np_img = clip[k].permute(1, 2, 0).contiguous().numpy()
        for axis in DEGRADE_ORDER:
            if axis in active_axes:
                np_img = per_frame_degs[axis].degrade(np_img, rng)
        out.append(torch.from_numpy(np_img.transpose(2, 0, 1)).contiguous())
    return torch.stack(out, dim=0)


def _apply_per_frame_single(
    clip: torch.Tensor, deg: Degradation, rng: random.Random,
) -> torch.Tensor:
    """Run a single Degradation over every frame (matches trainer._apply_per_frame_single)."""
    out = []
    for k in range(clip.shape[0]):
        np_img = clip[k].permute(1, 2, 0).contiguous().numpy()
        np_img = deg.degrade(np_img, rng)
        out.append(torch.from_numpy(np_img.transpose(2, 0, 1)).contiguous())
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------

class CompoundDegradationWrapper(Dataset):
    """Apply the trainer's per-batch degradation pipeline per-sample inside the DataLoader worker.

    The wrapper does *no* device transfer (workers run on CPU); the trainer
    is responsible for moving the resulting tensors to the device.

    Constructor accepts either a ``data_cfg`` object (anything with the
    same attribute names as :class:`restora_models.config.DataConfig`) or
    explicit kwargs. Explicit kwargs win over ``data_cfg`` when both are
    supplied.
    """

    def __init__(
        self,
        inner: Dataset,
        *,
        data_cfg: Any | None = None,
        film_overlay_root: str | Path | None = None,
        film_overlay_prob: float | None = None,
        film_color_cast_prob: float | None = None,
        gate_weave_prob: float | None = None,
        gate_weave_max_shift_px: float | None = None,
        mpeg_transcode_prob: float | None = None,
        seed: int = 0,
    ) -> None:
        self.inner = inner

        def _pick(explicit, attr, default):
            if explicit is not None:
                return explicit
            if data_cfg is not None and hasattr(data_cfg, attr):
                return getattr(data_cfg, attr)
            return default

        overlay_root = _pick(film_overlay_root, "film_overlay_root", None)
        self.film_overlay_prob = float(_pick(film_overlay_prob, "film_overlay_prob", 0.0))
        self.film_color_cast_prob = float(_pick(film_color_cast_prob, "film_color_cast_prob", 0.0))
        self.gate_weave_prob = float(_pick(gate_weave_prob, "gate_weave_prob", 0.0))
        self.gate_weave_max_shift_px = float(_pick(gate_weave_max_shift_px, "gate_weave_max_shift_px", 2.0))
        self.mpeg_prob = float(_pick(mpeg_transcode_prob, "mpeg_transcode_prob", 0.0))
        self.base_seed = int(seed)

        # Per-frame degradations (one Degradation per axis).
        self.per_frame_degs = _build_per_frame_degradations()

        # Optional film overlay — needs a real on-disk asset dir.
        self.film_overlay: FilmOverlayDegradation | None = None
        if overlay_root is not None:
            root = Path(overlay_root).expanduser()
            if root.exists() and any(root.rglob("*.png")):
                self.film_overlay = FilmOverlayDegradation.from_dir(root)
            else:
                # Loud-but-not-fatal: the trainer keeps running with the
                # synthetic axes only, but we want this state to be
                # obvious in the logs rather than silently dropped.
                _logger.warning(
                    "film_overlay_root=%s but no PNG textures found "
                    "there; film_overlay degradation will be SKIPPED. "
                    "Run `restora prepare-data film-overlays --out %s` "
                    "to populate (download or synthesize).",
                    overlay_root, overlay_root,
                )

        # Film color cast: per-frame, no asset dependency.
        self.film_color_cast = FilmColorCastDegradation()

        # Per-clip degradations.
        self.gate_weave = GateWeaveDegradation(max_shift_px=self.gate_weave_max_shift_px)
        # mpeg requires ffmpeg on PATH (see trainer.py:295).
        self.mpeg: MpegTranscodeDegradation | None = None
        if shutil.which("ffmpeg") is not None:
            self.mpeg = MpegTranscodeDegradation()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.inner)

    def _make_rng(self, idx: int) -> random.Random:
        """Build a per-sample RNG keyed by (worker_id, base_seed, idx).

        Using ``get_worker_info()`` keeps multiple workers from sampling
        identical sequences while remaining deterministic given a fixed
        ``seed`` + dataset ordering. When called from the main process
        (e.g. tests / num_workers=0) worker_id falls back to 0.
        """
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        # Mix via three large primes — avoids accidental cancellation
        # between worker_id and idx.
        key = (self.base_seed * 2_654_435_761
               + worker_id * 40_960_001
               + idx * 1_000_003)
        return random.Random(key & 0xFFFF_FFFF_FFFF_FFFF)

    def __getitem__(self, idx: int) -> dict:
        inner = self.inner[idx]
        frames = inner["frames"]  # (T, 3, H, W) float in [0,1]
        if not isinstance(frames, torch.Tensor):
            frames = torch.as_tensor(frames)

        rng = self._make_rng(idx)
        clean = frames.detach().cpu()

        active = _sample_axes(rng)
        clip = _apply_per_frame_degradations(clean, active, self.per_frame_degs, rng)

        if self.film_overlay is not None and rng.random() < self.film_overlay_prob:
            clip = _apply_per_frame_single(clip, self.film_overlay, rng)
        if rng.random() < self.film_color_cast_prob:
            clip = _apply_per_frame_single(clip, self.film_color_cast, rng)
        if self.gate_weave_prob > 0 and rng.random() < self.gate_weave_prob:
            clip = self.gate_weave.apply_clip(clip)
        if self.mpeg is not None and self.mpeg_prob > 0 and rng.random() < self.mpeg_prob:
            clip = self.mpeg.apply_clip(clip)

        config = _make_config_vec(active)
        axes_active = "+".join(sorted(active)) if active else "identity"
        return {
            "clean":       clean,
            "degraded":    clip.to(clean.dtype),
            "config":      config,
            "axes_active": axes_active,
        }


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def collate_compound(batch: list[dict]) -> dict:
    """Stack clean/degraded/config; keep axes_active as a list[str]."""
    return {
        "clean":       torch.stack([b["clean"]    for b in batch]),
        "degraded":    torch.stack([b["degraded"] for b in batch]),
        "config":      torch.stack([b["config"]   for b in batch]),
        "axes_active": [b["axes_active"]          for b in batch],
    }


# ---------------------------------------------------------------------------
# worker_init_fn
# ---------------------------------------------------------------------------

def compound_worker_init_fn(worker_id: int) -> None:
    """Optional ``worker_init_fn`` for the DataLoader.

    We already key our own ``random.Random`` per-sample via worker_id +
    base_seed + idx (see ``CompoundDegradationWrapper._make_rng``) so the
    only thing left is to make sure NumPy / Python global RNG state is
    distinct per worker in case a downstream degradation reaches for the
    global RNG. Without this, every worker inherits the same NumPy state
    from the parent (a PyTorch footgun).
    """
    import numpy as np
    info = get_worker_info()
    base = info.dataset.base_seed if (info is not None and hasattr(info.dataset, "base_seed")) else 0
    seed = (base * 2_654_435_761 + worker_id * 40_960_001) & 0xFFFF_FFFF
    np.random.seed(seed)
    random.seed(seed ^ 0x9E37_79B9)

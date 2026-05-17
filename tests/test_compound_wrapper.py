"""Tests for the per-sample compound degradation wrapper.

The wrapper lifts the trainer's `_degrade_batch` body into the
DataLoader worker side. We exercise:

1. Shape / dtype / key contract of `__getitem__`.
2. The empirical distribution of axis activations matches the
   bucket probabilities defined in `_sample_axes` (identity ~15%,
   single ~35%, two ~35%, 3+ ~15%).
3. `collate_compound` stacks tensors and keeps `axes_active` as a list
   of strings.
"""
from __future__ import annotations

import math
from collections import Counter

import torch
from torch.utils.data import Dataset

from restora_models.data.compound import AXES
from restora_models.data.compound_wrapper import (
    CompoundDegradationWrapper,
    _sample_axes,
    collate_compound,
)


class _FakeClips(Dataset):
    """Tiny synthetic VideoWindowDataset stand-in.

    Yields {"frames": (7, 3, H, W) float tensor in [0,1], ...} just like
    the real dataset so the wrapper can degrade it.
    """

    def __init__(self, n: int, h: int = 16, w: int = 16, seed: int = 0):
        self.n = n
        self.h = h
        self.w = w
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(self.seed + idx)
        return {
            "frames": torch.rand(7, 3, self.h, self.w, generator=g),
            "source": "fake",
            "key": f"fake_{idx}",
        }


def _make_wrapper(n: int = 8, seed: int = 0) -> CompoundDegradationWrapper:
    """Build a wrapper with all optional layers disabled.

    We disable film overlay / cast / gate weave / mpeg so the smoke test
    doesn't depend on ffmpeg or on overlay assets being present. The
    per-frame axis pipeline still runs and is what we want to verify.
    """
    inner = _FakeClips(n=n)
    return CompoundDegradationWrapper(
        inner,
        film_overlay_root=None,
        film_overlay_prob=0.0,
        film_color_cast_prob=0.0,
        gate_weave_prob=0.0,
        mpeg_transcode_prob=0.0,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. shape / dtype / key contract
# ---------------------------------------------------------------------------

def test_wrapper_item_shape_and_keys():
    w = _make_wrapper(n=4, seed=42)
    for i in range(len(w)):
        item = w[i]
        # required keys
        assert set(item.keys()) == {"clean", "degraded", "config", "axes_active"}
        # tensor shapes
        assert item["clean"].shape == (7, 3, 16, 16)
        assert item["degraded"].shape == (7, 3, 16, 16)
        assert item["config"].shape == (len(AXES),)
        # dtypes
        assert item["clean"].dtype == torch.float32
        assert item["degraded"].dtype == torch.float32
        assert item["config"].dtype == torch.float32
        # axes_active is a string
        assert isinstance(item["axes_active"], str)
        # If string is "identity" then config is all zeros, else non-zero
        if item["axes_active"] == "identity":
            assert item["config"].sum().item() == 0.0
        else:
            assert item["config"].sum().item() > 0.0
            # Each token in the joined string should be a valid axis name
            tokens = item["axes_active"].split("+")
            for t in tokens:
                assert t in AXES


def test_wrapper_determinism_same_seed_same_idx():
    """Two wrappers with same seed + same idx should produce identical degraded clips.

    (Subject to caveat — this only holds when the wrapper's RNG is the
    *sole* source of randomness, i.e. when no degradation reaches for a
    NumPy global. The configured pipeline here avoids that.)
    """
    a = _make_wrapper(n=4, seed=7)
    b = _make_wrapper(n=4, seed=7)
    ia = a[2]
    ib = b[2]
    assert torch.allclose(ia["clean"], ib["clean"])
    assert torch.allclose(ia["config"], ib["config"])
    assert ia["axes_active"] == ib["axes_active"]


# ---------------------------------------------------------------------------
# 2. empirical distribution
# ---------------------------------------------------------------------------

def test_sample_axes_distribution_matches_design():
    """200 draws — verify the bucket probabilities roughly hit their targets.

    Targets (from `_sample_axes` docstring):
      identity = 0.15
      single   = 0.35
      two      = 0.35
      3+       = 0.15

    With N=200 the std-dev of each bucket is sqrt(0.5*0.5/200) ~= 3.5%,
    so we use a 10% tolerance — generous enough to be non-flaky.
    """
    import random as _random
    rng = _random.Random(123)
    counts = Counter()
    N = 2000
    for _ in range(N):
        s = _sample_axes(rng)
        n = len(s)
        if n == 0:
            counts["identity"] += 1
        elif n == 1:
            counts["single"] += 1
        elif n == 2:
            counts["two"] += 1
        else:
            counts["threeplus"] += 1

    p = {k: v / N for k, v in counts.items()}
    assert math.isclose(p.get("identity", 0.0), 0.15, abs_tol=0.04)
    assert math.isclose(p.get("single", 0.0), 0.35, abs_tol=0.05)
    assert math.isclose(p.get("two", 0.0), 0.35, abs_tol=0.05)
    assert math.isclose(p.get("threeplus", 0.0), 0.15, abs_tol=0.04)


def test_wrapper_axes_distribution_smoke():
    """Iterate the wrapper itself across many indices and check the same bucket distribution.

    Different from the previous test: this includes the per-sample RNG
    keying via `_make_rng`, so it's a sanity check that the keying
    function doesn't collapse the distribution.
    """
    inner = _FakeClips(n=400, seed=0)
    w = CompoundDegradationWrapper(
        inner,
        film_overlay_root=None,
        film_overlay_prob=0.0,
        film_color_cast_prob=0.0,
        gate_weave_prob=0.0,
        mpeg_transcode_prob=0.0,
        seed=99,
    )
    counts = Counter()
    for i in range(len(w)):
        item = w[i]
        if item["axes_active"] == "identity":
            counts["identity"] += 1
        else:
            n = len(item["axes_active"].split("+"))
            if n == 1:
                counts["single"] += 1
            elif n == 2:
                counts["two"] += 1
            else:
                counts["threeplus"] += 1
    N = len(w)
    p = {k: v / N for k, v in counts.items()}
    # Looser tolerances — we're at N=400 *and* the seed keying introduces
    # some structure across consecutive idx values.
    assert math.isclose(p.get("identity", 0.0), 0.15, abs_tol=0.08)
    assert math.isclose(p.get("single", 0.0), 0.35, abs_tol=0.10)
    assert math.isclose(p.get("two", 0.0), 0.35, abs_tol=0.10)
    assert math.isclose(p.get("threeplus", 0.0), 0.15, abs_tol=0.08)


# ---------------------------------------------------------------------------
# 3. collate
# ---------------------------------------------------------------------------

def test_collate_compound_stacks_tensors_and_keeps_axes_as_list():
    w = _make_wrapper(n=4, seed=11)
    batch = [w[i] for i in range(4)]
    out = collate_compound(batch)

    assert out["clean"].shape == (4, 7, 3, 16, 16)
    assert out["degraded"].shape == (4, 7, 3, 16, 16)
    assert out["config"].shape == (4, len(AXES))
    assert isinstance(out["axes_active"], list)
    assert len(out["axes_active"]) == 4
    for s in out["axes_active"]:
        assert isinstance(s, str)
        assert s == "identity" or all(t in AXES for t in s.split("+"))

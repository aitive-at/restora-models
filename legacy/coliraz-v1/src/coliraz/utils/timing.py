"""Tiny EMA + Stopwatch for the trainer/UI."""
from __future__ import annotations

import time


class EMA:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1 - self.alpha) * self.value
        return self.value


class Stopwatch:
    def __init__(self) -> None:
        self._t0: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        assert self._t0 is not None
        self.elapsed = time.perf_counter() - self._t0
        return self.elapsed

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

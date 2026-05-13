"""Degradation registry. Each registered Degradation has a name and
a degrade() method that maps (H, W, 3) float32 RGB → degraded same-shape RGB."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Type

import numpy as np


class Degradation(ABC):
    name: str = ""
    task_id: int = -1  # assigned by config at startup

    @abstractmethod
    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        """rgb: (H, W, 3) float32 in [0, 1]. Returns same-shape degraded RGB."""


DEGRADATION_REGISTRY: dict[str, Type[Degradation]] = {}


def register_degradation(name: str):
    def deco(cls: Type[Degradation]):
        if name in DEGRADATION_REGISTRY:
            raise KeyError(f"degradation {name!r} already registered")
        cls.name = name
        DEGRADATION_REGISTRY[name] = cls
        return cls

    return deco


def build_degradation(name: str, cfg: dict[str, Any] | None = None) -> Degradation:
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"unknown degradation {name!r}; have {sorted(DEGRADATION_REGISTRY)}")
    cfg = dict(cfg or {})
    return DEGRADATION_REGISTRY[name](**cfg)

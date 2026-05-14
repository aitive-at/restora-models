"""Model registry: future architectures plug in via @register_model with
the same forward contract (rgb, task) -> rgb."""
from __future__ import annotations

from typing import Type

from torch import nn

from restora_models.config import ModelConfig

MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    def deco(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            raise KeyError(f"model {name!r} already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


def build_model(cfg: ModelConfig, *, num_axes: int = 5) -> nn.Module:
    if cfg.type not in MODEL_REGISTRY:
        raise KeyError(f"unknown model type {cfg.type!r}; have {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[cfg.type](cfg, num_axes=num_axes)

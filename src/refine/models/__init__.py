"""Refine models. Importing this module registers all backbones."""
from . import nafnet as _nafnet  # noqa: F401
from . import promptir as _promptir  # noqa: F401  registers "promptir"
from .registry import MODEL_REGISTRY, build_model, register_model

__all__ = ["MODEL_REGISTRY", "build_model", "register_model"]

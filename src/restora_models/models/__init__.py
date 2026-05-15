"""Restora models. Importing this module registers the backbone."""
from . import nafnet as _nafnet  # noqa: F401
from .registry import MODEL_REGISTRY, build_model, register_model

__all__ = ["MODEL_REGISTRY", "build_model", "register_model"]

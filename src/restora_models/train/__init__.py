"""Training entry points.

Trainer and fit are imported lazily; direct callers should use
``from restora_models.train.trainer import Trainer`` for backwards
compatibility. The trainer is being rewritten in Phase 10 of the
temporal redesign, so importing eagerly here would break unrelated
test collection during the build.
"""

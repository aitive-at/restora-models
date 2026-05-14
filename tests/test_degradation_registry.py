import numpy as np
import pytest

from restora_models.data.degradations.registry import (
    DEGRADATION_REGISTRY,
    Degradation,
    build_degradation,
    register_degradation,
)


def test_registry_collects_decorated_class():
    @register_degradation("toy_test")
    class _Toy(Degradation):
        name = "toy_test"

        def __init__(self, weight: float = 1.0):
            super().__init__()
            self.weight = weight

        def degrade(self, rgb, rng):
            return rgb

    assert "toy_test" in DEGRADATION_REGISTRY
    d = build_degradation("toy_test", {"weight": 2.0})
    assert d.weight == 2.0
    assert d.degrade(np.zeros((4, 4, 3), dtype=np.float32), None).shape == (4, 4, 3)
    DEGRADATION_REGISTRY.pop("toy_test")


def test_build_unknown_raises():
    with pytest.raises(KeyError):
        build_degradation("nope", {})

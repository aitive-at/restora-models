import torch

from restora_models.data.compound import AXES, CompoundDegradationWrapper, collate_compound


class _DummyCleanDS:
    def __len__(self): return 8
    def __getitem__(self, idx):
        return torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(idx))


def test_compound_wrapper_returns_expected_keys():
    ds = _DummyCleanDS()
    wrap = CompoundDegradationWrapper(ds, axis_probs={a: 0.5 for a in AXES}, seed=0)
    sample = wrap[0]
    assert set(sample.keys()) == {"clean", "degraded", "config", "axes"}
    assert sample["clean"].shape == (3, 32, 32)
    assert sample["degraded"].shape == (3, 32, 32)
    assert sample["config"].shape == (5,)
    assert isinstance(sample["axes"], str)


def test_compound_identity_case_has_zero_config():
    ds = _DummyCleanDS()
    # identity_prob=1.0 forces identity on every sample
    wrap = CompoundDegradationWrapper(ds, axis_probs={a: 0.5 for a in AXES},
                                       identity_prob=1.0, seed=0)
    sample = wrap[0]
    assert (sample["config"] == 0).all()
    assert sample["axes"] == "identity"
    # degraded should equal clean for identity
    torch.testing.assert_close(sample["clean"], sample["degraded"])


def test_compound_random_subset_is_deterministic_for_same_idx():
    ds = _DummyCleanDS()
    wrap = CompoundDegradationWrapper(ds, axis_probs={a: 0.5 for a in AXES}, seed=42)
    s1 = wrap[3]
    s2 = wrap[3]
    assert s1["axes"] == s2["axes"]
    torch.testing.assert_close(s1["config"], s2["config"])
    torch.testing.assert_close(s1["degraded"], s2["degraded"])


def test_collate_stacks():
    ds = _DummyCleanDS()
    wrap = CompoundDegradationWrapper(ds, axis_probs={a: 0.5 for a in AXES}, seed=0)
    batch = collate_compound([wrap[i] for i in range(4)])
    assert batch["clean"].shape == (4, 3, 32, 32)
    assert batch["degraded"].shape == (4, 3, 32, 32)
    assert batch["config"].shape == (4, 5)
    assert len(batch["axes"]) == 4
    assert all(isinstance(s, str) for s in batch["axes"])

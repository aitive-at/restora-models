from refine.config import ModelConfig


def test_promptir_fields_default_to_none():
    m = ModelConfig(type="promptir", size="large")
    assert m.prompt_n is None
    assert m.prompt_dim is None
    assert m.prompt_hw is None


def test_promptir_fields_accept_int_overrides():
    m = ModelConfig(type="promptir", size="tiny", prompt_n=7,
                    prompt_dim=48, prompt_hw=8)
    assert m.prompt_n == 7
    assert m.prompt_dim == 48
    assert m.prompt_hw == 8


def test_nafnet_unaffected():
    m = ModelConfig(type="nafnet", size="tiny", nf=32)
    assert m.nf == 32
    assert m.prompt_n is None

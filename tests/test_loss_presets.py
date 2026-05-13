def test_standard_preset_includes_chroma_lab():
    from refine.config import expand_loss_preset
    losses = expand_loss_preset("standard")
    names = [l.name for l in losses]
    assert "chroma_lab" in names
    # chroma_lab weight is moderate (raw loss is in ab-L1 units, ~10-30 per batch)
    chroma = [l for l in losses if l.name == "chroma_lab"][0]
    assert 0.001 <= chroma.weight <= 0.2, f"chroma_lab weight out of range: {chroma.weight}"
    # colorfulness weight reduced
    cf = [l for l in losses if l.name == "colorfulness"]
    if cf:
        assert cf[0].weight <= 0.05, f"colorfulness weight too high: {cf[0].weight}"


def test_vivid_preset_keeps_chroma_anchor():
    from refine.config import expand_loss_preset
    losses = expand_loss_preset("vivid")
    names = [l.name for l in losses]
    assert "chroma_lab" in names


def test_full_preset_includes_chroma_lab():
    from refine.config import expand_loss_preset
    losses = expand_loss_preset("full")
    names = [l.name for l in losses]
    assert "chroma_lab" in names

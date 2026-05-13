import pytest
import torch

from coliraz.models.encoder import ConvNeXtEncoder


@pytest.mark.parametrize("size,expected_channels", [
    ("tiny", [96, 192, 384, 768]),
    ("large", [192, 384, 768, 1536]),
])
def test_encoder_returns_four_features(size, expected_channels):
    enc = ConvNeXtEncoder(size=size, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    feats = enc(x)
    assert len(feats) == 4
    assert [f.shape[1] for f in feats] == expected_channels
    spat = [f.shape[2] for f in feats]
    assert spat[0] > spat[1] > spat[2] > spat[3]


def test_encoder_feature_channels_property():
    enc = ConvNeXtEncoder(size="tiny", pretrained=False)
    assert enc.feature_channels == [96, 192, 384, 768]

"""Tests for feature-matching distillation loss."""
import torch

from restora_models.losses.feat_match import FeatureMatchLoss


def test_feat_match_zero_for_matching():
    loss = FeatureMatchLoss()
    feats = [torch.rand(2, 32, 16, 16), torch.rand(2, 64, 8, 8)]
    val = loss(feats, [f.clone() for f in feats])
    assert val.item() < 1e-6


def test_feat_match_positive_for_mismatched():
    loss = FeatureMatchLoss()
    a = [torch.rand(2, 32, 16, 16)]
    b = [a[0] + 0.5]
    val = loss(a, b)
    assert val.item() > 0.01

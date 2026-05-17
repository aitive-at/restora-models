"""Feature-matching loss for SLKD-style distillation.

Takes two parallel lists of teacher / student features (tuples of
(B, C, H, W) tensors at matching decoder stages) and returns mean MSE.
Used in Phase 14 distillation; not part of the main loss aggregator.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchLoss(nn.Module):
    def forward(self, teacher_feats: Sequence[torch.Tensor],
                student_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(teacher_feats) != len(student_feats):
            raise ValueError(
                f"teacher/student feature lengths differ: "
                f"{len(teacher_feats)} vs {len(student_feats)}"
            )
        total = teacher_feats[0].new_zeros(())
        for t, s in zip(teacher_feats, student_feats):
            if t.shape[1] != s.shape[1]:
                if s.shape[1] > t.shape[1]:
                    s = s[:, :t.shape[1]]
                else:
                    pad = t.new_zeros(s.shape[0], t.shape[1] - s.shape[1], *s.shape[2:])
                    s = torch.cat([s, pad], dim=1)
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
            total = total + F.mse_loss(s, t)
        return total / max(len(teacher_feats), 1)

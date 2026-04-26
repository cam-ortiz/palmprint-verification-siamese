"""
Loss functions for Siamese palmprint verification.
"""

from __future__ import annotations

import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.

    Labels:
        1 = genuine pair, same identity
        0 = impostor pair, different identity

    For genuine pairs:
        minimize distance

    For impostor pairs:
        push distance above margin
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        distance: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        label = label.float()

        genuine_loss = label * torch.pow(distance, 2)

        impostor_loss = (1.0 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0),
            2,
        )

        loss = torch.mean(genuine_loss + impostor_loss)

        return loss
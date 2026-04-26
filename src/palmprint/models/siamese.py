"""
Siamese network for palmprint verification.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from palmprint.models.backbone import PalmprintCNNBackbone


class SiameseNetwork(nn.Module):
    """
    Siamese model using a shared CNN backbone.

    Input:
        img1, img2

    Output:
        embedding1, embedding2, distance
    """

    def __init__(self, embedding_dim: int = 128, input_channels: int = 3):
        super().__init__()

        self.backbone = PalmprintCNNBackbone(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
        )

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb1 = self.backbone(img1)
        emb2 = self.backbone(img2)

        distance = F.pairwise_distance(emb1, emb2)

        return emb1, emb2, distance
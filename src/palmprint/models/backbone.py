"""
CNN backbone for palmprint embedding extraction.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PalmprintCNNBackbone(nn.Module):
    """
    Small CNN that converts an ROI image into an embedding vector.
    """

    def __init__(self, embedding_dim: int = 128, input_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.embedding(x)

        # Normalize embeddings so distance comparisons are more stable.
        x = F.normalize(x, p=2, dim=1)

        return x
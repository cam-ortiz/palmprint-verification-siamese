from __future__ import annotations

from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset


Pair = Tuple[str, str, int]


class PairDataset(Dataset):
    """
    Dataset for Siamese training.

    Returns:
        img1_tensor, img2_tensor, label
    """

    def __init__(
        self,
        pairs: List[Pair],
        image_size: int = 224,
        grayscale: bool = True,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)

        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")

        # Convert to grayscale if enabled
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.image_size, self.image_size))

            # shape: (H, W) -> (1, H, W)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))

            # shape: (H, W, 3) -> (3, H, W)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # Normalize to [0, 1]
        img = img / 255.0

        return img

    def __getitem__(self, idx: int):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label_tensor


"""
Training utilities for Siamese palmprint verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the Siamese model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for img1, img2, labels in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        _emb1, _emb2, distances = model(img1, img2)
        loss = criterion(distances, labels)

        loss.backward()
        optimizer.step()

        batch_size = img1.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Validate the Siamese model for one epoch.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for img1, img2, labels in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device).float()

        _emb1, _emb2, distances = model(img1, img2)
        loss = criterion(distances, labels)

        batch_size = img1.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_siamese_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    checkpoint_path: str | Path,
) -> dict[str, list[float]]:
    """
    Full training loop.

    Saves the best model based on validation loss.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                checkpoint_path,
            )

            print(f"Saved best model to {checkpoint_path}")

    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load a saved model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return checkpoint
"""
Dataset utilities for the Birjand Mobile Palmprint Database (BMPD).

This module treats each person's left and right palm as separate verification
identities. For example:

    023_F_L_33.png -> identity 023_L
    023_F_R_34.png -> identity 023_R

This avoids training the model to treat left and right palms from the same
person as the same biometric identity.
"""


from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict


@dataclass(frozen=True)
class Sample:
    img_path: str
    label: int
    subject_id: str      # Example: "023_L"
    person_id: str       # Example: "023"
    hand: str            # "L" or "R"


VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heif", ".heic"
}


def parse_bmpd_filename(path: str | Path) -> tuple[str, str]:
    """
    Parse BMPD filename and return person_id and hand side.

    Expected example:
        023_F_L_33.png

    Returns:
        person_id = "023"
        hand = "L"
    """
    stem = Path(path).stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected BMPD filename format: {path}")

    person_id = parts[0]
    hand = parts[2].upper()

    if hand not in {"L", "R"}:
        raise ValueError(f"Could not parse hand side from filename: {path}")

    return person_id, hand


def collect_bmpd_samples(root_dir: str | Path) -> list[Sample]:
    """
    Collect all BMPD image samples.

    This expects a structure like:

        data/processed/bmpd_roi/
            001/
            002/
            003/

    Labels are created from filename person + hand side, not just
    folder name.
    """
    root_dir = Path(root_dir)

    image_paths = sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    identity_to_paths: dict[str, list[Path]] = defaultdict(list)

    for image_path in image_paths:
        person_id, hand = parse_bmpd_filename(image_path)
        subject_id = f"{person_id}_{hand}"
        identity_to_paths[subject_id].append(image_path)

    subject_ids = sorted(identity_to_paths.keys())
    subject_to_label = {subject_id: i for i, subject_id in enumerate(subject_ids)}

    samples: list[Sample] = []

    for subject_id in subject_ids:
        person_id, hand = subject_id.split("_")
        label = subject_to_label[subject_id]

        for image_path in identity_to_paths[subject_id]:
            samples.append(
                Sample(
                    img_path=str(image_path),
                    label=label,
                    subject_id=subject_id,
                    person_id=person_id,
                    hand=hand,
                )
            )

    return samples


def split_subject_ids(
    subject_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split identities into train/validation/test sets.
    This splits by identity, not by image.
    Since identity is person + hand, 023_L and 023_R are separate classes.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    subject_ids = subject_ids.copy()
    random.Random(seed).shuffle(subject_ids)

    n_total = len(subject_ids)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:n_train + n_val]
    test_ids = subject_ids[n_train + n_val:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def build_bmpd_splits(
    root_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict[str, list[Sample]]:
    """
    Build BMPD train/validation/test splits.

    The split is performed by person-hand identity, not by image.

    Example identities:
        001_L
        001_R
        002_L
        002_R
    """
    samples = collect_bmpd_samples(root_dir)

    subject_ids = sorted({sample.subject_id for sample in samples})
    split_ids = split_subject_ids(
        subject_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    splits: dict[str, list[Sample]] = {}

    for split_name, ids in split_ids.items():
        id_set = set(ids)
        splits[split_name] = [
            sample for sample in samples if sample.subject_id in id_set
        ]

    return splits
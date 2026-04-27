"""
Utilities for filtering ROI images using manual quality-control annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_bad_roi_paths(
    quality_csv_path: Path,
    project_root: Path,
    bad_label: str = "b",
) -> set[Path]:
    """
    Load paths for ROI images marked as bad.

    Parameters
    ----------
    quality_csv_path : Path
        Path to the ROI quality CSV.
    project_root : Path
        Root directory of the project.
    bad_label : str, optional
        Label used to mark bad images.

    Returns
    -------
    set[Path]
        Set of absolute paths for images that should be excluded.
    """
    if not quality_csv_path.exists():
        return set()

    df = pd.read_csv(quality_csv_path)

    bad_df = df[
        df["quality"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .eq(bad_label)
    ]

    return {
        (project_root / path).resolve()
        for path in bad_df["path"]
    }



def filter_bad_roi_paths(
    image_paths: list[Path],
    quality_csv_path: Path,
    project_root: Path,
    verbose: bool = True,
) -> Tuple[list[Path], dict]:
    """
    Filter out ROI images marked as bad and return stats.
    """
    image_paths = [p.resolve() for p in image_paths]

    bad_paths = load_bad_roi_paths(
        quality_csv_path=quality_csv_path,
        project_root=project_root,
    )

    filtered_paths = [
        path for path in image_paths
        if path not in bad_paths
    ]

    total = len(image_paths)
    removed = total - len(filtered_paths)
    percent = (removed / total * 100) if total > 0 else 0.0

    stats = {
        "total": total,
        "removed": removed,
        "kept": len(filtered_paths),
        "percent_removed": percent,
    }

    if verbose:
        print("\n=== ROI QUALITY FILTERING ===")
        print(f"Total images:     {total}")
        print(f"Removed (bad):    {removed}")
        print(f"Kept:             {len(filtered_paths)}")
        print(f"Percent removed:  {percent:.2f}%")

    return filtered_paths, stats
"""
Create a starter ROI quality CSV for manual QC annotation.

The CSV is ordered the same way as the contact sheets:
by the final image number in the ROI filename.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def roi_filename_sort_key(path: Path) -> tuple[str, int, str]:
    """
    Sort ROI filenames by subject first, then final image number.

    Example
    -------
    001_F_L_30_roi.png -> subject 001, image number 30
    001_S_R_9_roi.png  -> subject 001, image number 9
    """
    subject_id = path.name.split("_")[0]

    match = re.search(r"_(\d+)_roi\.png$", path.name)
    image_number = int(match.group(1)) if match else 999999

    return (subject_id, image_number, path.name)


def parse_roi_filename(path: Path) -> dict[str, str]:
    """
    Parse metadata from an ROI filename.

    Expected filename format:
    subject_session_hand_number_roi.png

    Example
    -------
    001_F_L_30_roi.png
    """
    parts = path.stem.split("_")

    if len(parts) < 5:
        raise ValueError(f"Unexpected ROI filename format: {path.name}")

    subject_id = parts[0]
    hand = parts[2]

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "subject_id": subject_id,
        "hand": hand,
        "quality": "",
        "reason": "",
    }


def create_roi_quality_template(
    roi_dir: Path,
    output_path: Path,
    image_pattern: str = "*_roi.png",
) -> None:
    """
    Create a starter CSV with one row per ROI image.
    """
    image_paths = sorted(
        roi_dir.rglob(image_pattern),
        key=roi_filename_sort_key,
    )

    if not image_paths:
        raise FileNotFoundError(f"No ROI images found in: {roi_dir}")

    rows = [parse_roi_filename(path) for path in image_paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["path", "subject_id", "hand", "quality", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved ROI quality template: {output_path}")
    print(f"Rows written: {len(rows)}")


def main() -> None:
    roi_dir = PROJECT_ROOT / "data" / "processed" / "bmpd_roi"
    output_path = PROJECT_ROOT / "data" / "qc" / "roi_quality_template.csv"

    create_roi_quality_template(
        roi_dir=roi_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
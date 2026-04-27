"""
Create ROI contact sheets for manual quality control review.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from palmprint.preprocessing.contact_sheet import create_subject_contact_sheets


def main() -> None:
    roi_dir = PROJECT_ROOT / "data" / "processed" / "bmpd_roi"
    output_dir = PROJECT_ROOT / "data" / "qc" / "contact_sheets"

    create_subject_contact_sheets(
        roi_dir=roi_dir,
        output_dir=output_dir,
        image_pattern="*_roi.png",
        thumbnail_size=(128, 128),
        columns=6,
    )


if __name__ == "__main__":
    main()
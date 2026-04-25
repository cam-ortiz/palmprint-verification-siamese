"""
Preprocess BMPD palm images by extracting and saving final ROI crops.

Example:
    python scripts/preprocess_bmpd_roi.py \
        --input-root data/raw/bmpd \
        --output-root data/processed/bmpd_roi \
        --roi-size 224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from palmprint.preprocessing.roi import RoiExtractionConfig, extract_hand_roi


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_final_roi_config() -> RoiExtractionConfig:
    """
    Frozen ROI configuration for the final project pipeline.
    """
    return RoiExtractionConfig(
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        blur_ksize=(7, 7),
        threshold_method="otsu",
        morph_kernel_size=(11, 11),
        morph_close_iterations=2,
        morph_open_iterations=1,
        morph_close_first=True,
        bbox_margin_frac=0.03,
        roi_crop_frac=0.55,
        crop_method="centroid",
        rotate_to_principal_axis=True,
        rotation_center="centroid",
    )


def preprocess_bmpd_roi(
    input_root: Path,
    output_root: Path,
    roi_size: int = 224,
    overwrite: bool = False,
) -> None:
    """
    Apply ROI extraction to every BMPD image and save resized ROI crops.

    Expected input structure:
        data/raw/bmpd/BMPD/
            001/
            002/
            003/

    Output structure:
        data/processed/bmpd_roi/
            001/
            002/
            003/
    """
    config = build_final_roi_config()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    subject_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

    if not subject_dirs:
        raise RuntimeError(f"No subject folders found in: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_failed = 0

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        subject_output_dir = output_root / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in subject_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )

        print(f"Processing subject {subject_id}: {len(image_paths)} images")

        for image_path in image_paths:
            output_path = subject_output_dir / f"{image_path.stem}_roi.png"

            if output_path.exists() and not overwrite:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"  Failed to read image: {image_path}")
                total_failed += 1
                continue

            try:
                result = extract_hand_roi(image, config)
                roi = result.final_roi.image

                roi_resized = cv2.resize(
                    roi,
                    (roi_size, roi_size),
                    interpolation=cv2.INTER_AREA,
                )

                cv2.imwrite(str(output_path), roi_resized)
                total_saved += 1

            except Exception as exc:
                print(f"  Failed ROI extraction for {image_path}: {exc}")
                total_failed += 1

    print()
    print("Finished ROI preprocessing")
    print(f"Saved ROI images: {total_saved}")
    print(f"Failed images: {total_failed}")
    print(f"Output folder: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and save BMPD palm ROI images."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/raw/bmpd/BMPD"),
        help="Path to raw BMPD subject folders.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/bmpd_roi"),
        help="Path where processed ROI images will be saved.",
    )

    parser.add_argument(
        "--roi-size",
        type=int,
        default=224,
        help="Final square ROI image size.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed ROI images.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preprocess_bmpd_roi(
        input_root=args.input_root,
        output_root=args.output_root,
        roi_size=args.roi_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

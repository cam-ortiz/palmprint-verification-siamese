"""
Utilities for creating ROI contact sheets for manual quality review.

These contact sheets make it easier to visually inspect processed palm ROI
images and identify bad crops before training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

from PIL import Image, ImageDraw, ImageFont


def roi_filename_sort_key(path: Path) -> tuple[int, str]:
    """
    Sort ROI filenames by the final image number.

    Example
    -------
    001_F_L_30_roi.png -> 30
    001_S_R_9_roi.png  -> 9
    """
    match = re.search(r"_(\d+)_roi\.png$", path.name)

    if match is None:
        return (999999, path.name)

    image_number = int(match.group(1))
    return (image_number, path.name)


def get_subject_id_from_filename(image_path: Path) -> str:
    """
    Extract the subject ID from a processed ROI filename.

    Example
    -------
    020_F_L_30_roi.png -> 020

    Parameters
    ----------
    image_path : Path
        Path to an ROI image.

    Returns
    -------
    str
        Subject ID parsed from the filename.
    """
    return image_path.stem.split("_")[0]


def group_images_by_subject(image_paths: Iterable[Path]) -> dict[str, list[Path]]:
    """
    Group ROI image paths by subject ID.

    Parameters
    ----------
    image_paths : Iterable[Path]
        ROI image paths.

    Returns
    -------
    dict[str, list[Path]]
        Mapping of subject ID to ROI image paths.
    """
    grouped: dict[str, list[Path]] = {}

    for image_path in image_paths:
        subject_id = get_subject_id_from_filename(image_path)
        grouped.setdefault(subject_id, []).append(image_path)

    return grouped


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    thumbnail_size: tuple[int, int] = (128, 128),
    columns: int = 6,
    padding: int = 12,
    label_height: int = 34,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Create a single contact sheet from ROI images.

    Parameters
    ----------
    image_paths : list[Path]
        Paths to ROI images.
    output_path : Path
        Where to save the generated contact sheet.
    thumbnail_size : tuple[int, int], optional
        Size of each thumbnail.
    columns : int, optional
        Number of columns in the contact sheet.
    padding : int, optional
        Space between thumbnails.
    label_height : int, optional
        Space reserved under each thumbnail for the filename.
    background_color : tuple[int, int, int], optional
        RGB background color.
    """
    if not image_paths:
        raise ValueError("image_paths must contain at least one image.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_paths, key=roi_filename_sort_key)

    rows = (len(image_paths) + columns - 1) // columns

    cell_width = thumbnail_size[0] + padding
    cell_height = thumbnail_size[1] + label_height + padding

    sheet_width = columns * cell_width + padding
    sheet_height = rows * cell_height + padding

    sheet = Image.new("RGB", (sheet_width, sheet_height), background_color)
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    for index, image_path in enumerate(image_paths):
        row = index // columns
        column = index % columns

        x = padding + column * cell_width
        y = padding + row * cell_height

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(thumbnail_size)

            thumb_x = x + (thumbnail_size[0] - img.width) // 2
            thumb_y = y

            sheet.paste(img, (thumb_x, thumb_y))

        label = image_path.name
        if len(label) > 24:
            label = label[:21] + "..."

        draw.text(
            (x, y + thumbnail_size[1] + 4),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    sheet.save(output_path)


def create_subject_contact_sheets(
    roi_dir: Path,
    output_dir: Path,
    image_pattern: str = "*_roi.png",
    thumbnail_size: tuple[int, int] = (128, 128),
    columns: int = 6,
) -> None:
    """
    Create one contact sheet per subject.

    Parameters
    ----------
    roi_dir : Path
        Directory containing processed ROI images.
    output_dir : Path
        Directory where contact sheets should be saved.
    image_pattern : str, optional
        Glob pattern used to find ROI images.
    thumbnail_size : tuple[int, int], optional
        Size of each thumbnail.
    columns : int, optional
        Number of columns in each contact sheet.
    """
    roi_dir = Path(roi_dir)
    output_dir = Path(output_dir)

    image_paths = list(roi_dir.rglob(image_pattern))

    if not image_paths:
        raise FileNotFoundError(f"No ROI images found in: {roi_dir}")

    grouped_images = group_images_by_subject(image_paths)

    for subject_id, subject_images in sorted(grouped_images.items()):
        output_path = output_dir / f"{subject_id}_contact_sheet.png"

        create_contact_sheet(
            image_paths=subject_images,
            output_path=output_path,
            thumbnail_size=thumbnail_size,
            columns=columns,
        )

        print(f"Saved contact sheet: {output_path}")# -*- coding: utf-8 -*-


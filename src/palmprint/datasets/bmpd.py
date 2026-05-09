"""
Dataset utilities for the Birjand Mobile Palmprint Database (BMPD).

This module provides helper functions for collecting BMPD palmprint images,
parsing BMPD filenames, filtering low-quality ROI images, and building
train/validation/test splits for palmprint verification experiments.

Each person's left and right palm are treated as separate biometric identities.
For example:

    023_F_L_33.png -> identity_id = "023_L"
    023_F_R_34.png -> identity_id = "023_R"

This prevents the model from learning that left and right palms from the same
person should be treated as the same verification identity.
"""


from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from palmprint.datasets.roi_quality import filter_bad_roi_paths


VALID_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".heif",
    ".heic",
}


@dataclass(frozen=True)
class BMPDSample:
    """
    Represents one BMPD palmprint image sample.

    Parameters
    ----------
    img_path : str
        Absolute or project-relative path to the palmprint image.
    label : int
        Integer class label assigned to the biometric identity.
    identity_id : str
        Person-hand identity identifier, such as ``"023_L"``.
    person_id : str
        Original BMPD person identifier, such as ``"023"``.
    hand : str
        Hand side. Expected values are ``"L"`` or ``"R"``.

    Notes
    -----
    The ``identity_id`` intentionally combines person and hand. This means
    ``023_L`` and ``023_R`` are treated as different biometric identities.
    """
    img_path: str
    label: int
    identity_id: str
    person_id: str
    hand: str


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for splitting dataset identities into train, validation, and
    test sets.

    Parameters
    ----------
    train_ratio : float, default=0.7
        Proportion of identities assigned to the training split.
    val_ratio : float, default=0.1
        Proportion of identities assigned to the validation split.
    test_ratio : float, default=0.2
        Proportion of identities assigned to the test split.
    seed : int, default=42
        Random seed used to shuffle identities before splitting.

    Notes
    -----
    These ratios are intended to be applied at the identity level, not the 
    image level. For BMPD, this means all images for an identity such as 
    ``023_L`` should stay within the same split.
    """
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class QualityFilterConfig:
    """
    Configuration for filtering low-quality BMPD ROI samples.

    Parameters
    ----------
    quality_csv_path : pathlib.Path, optional
        Path to the ROI quality CSV file. If omitted, the dataset builder may
        infer the path from ``project_root``.
    project_root : pathlib.Path, optional
        Root directory of the project repository. This is used to resolve paths
        during ROI quality filtering. If omitted, the dataset builder may infer
        it from the configured ROI directory.
    apply_quality_filter : bool, default=True
        Whether to remove known bad ROI samples before creating dataset
        samples.
    verbose : bool, default=True
        Whether to print ROI quality filtering information.

    Notes
    -----
    Set ``apply_quality_filter`` to ``False`` when you want to collect all ROI
    images without excluding entries listed in the quality-control CSV.
    """
    quality_csv_path: Path | None = None
    project_root: Path | None = None
    apply_quality_filter: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class BMPDDatasetConfig:
    """
    Top-level configuration for building BMPD dataset samples and splits.

    Parameters
    ----------
    roi_dir : pathlib.Path
        Directory containing processed BMPD ROI images.
    split : SplitConfig, default=SplitConfig()
        Configuration controlling train, validation, and test split behavior.
    quality_filter : QualityFilterConfig, default=QualityFilterConfig()
        Configuration controlling ROI quality filtering behavior.

    Notes
    -----
    This config groups dataset paths, split settings, and quality-filtering
    settings into one object so they can be passed cleanly into a dataset
    builder.

    Examples
    --------
    >>> config = BMPDDatasetConfig(
    ...     roi_dir=Path("data/processed/bmpd_roi"),
    ...     split=SplitConfig(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2),
    ...     quality_filter=QualityFilterConfig(apply_quality_filter=True),
    ... )
    """
    roi_dir: Path
    split: SplitConfig = field(default_factory=SplitConfig)
    quality_filter: QualityFilterConfig = field(
        default_factory=QualityFilterConfig
    )



class BMPDDatasetBuilder:
    """
    Builds BMPD samples and train/validation/test splits.

    Parameters
    ----------
    config : BMPDDatasetConfig
        Dataset configuration containing paths, split ratios, random seed, and
        quality filtering options.

    Notes
    -----
    This class is useful when the same BMPD configuration is reused across
    multiple scripts, notebooks, or training workflows.
    """

    def __init__(self, config: BMPDDatasetConfig) -> None:
        """
        Initialize the BMPD dataset builder.

        Parameters
        ----------
        config : BMPDDatasetConfig
            Configuration used for sample collection and split creation.
        """
        self.config = config


    @staticmethod
    def parse_bmpd_filename(path: str | Path) -> tuple[str, str]:
        """
        Parse a BMPD filename and return the person ID and hand side.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a BMPD image file. The filename is expected to follow a
            pattern similar to ``023_F_L_33.png``.

        Returns
        -------
        person_id : str
            Person identifier parsed from the filename, such as ``"023"``.
        hand : str
            Hand side parsed from the filename. Expected values are ``"L"`` or
            ``"R"``.

        Raises
        ------
        ValueError
            If the filename does not contain enough parts or if the hand side
            cannot be parsed.
        """
        stem = Path(path).stem
        parts = stem.split("_")

        if len(parts) < 3:
            raise ValueError(
                f"Unexpected BMPD filename format: {path}. "
                "Expected a name similar to '023_F_L_33.png'."
            )

        person_id = parts[0]
        hand = parts[2].upper()

        if hand not in {"L", "R"}:
            raise ValueError(
                f"Could not parse hand side from filename: {path}. "
                "Expected hand side to be 'L' or 'R'."
            )

        return person_id, hand


    def _collect_image_paths(self) -> list[Path]:
        """
        Recursively collect valid image paths from the configured BMPD root.

        Returns
        -------
        list of pathlib.Path
            Sorted list of image file paths.

        Raises
        ------
        FileNotFoundError
            If the BMPD root directory does not exist.
        NotADirectoryError
            If the BMPD root path is not a directory.
        """
        roi_dir = self.config.roi_dir.resolve()

        if not roi_dir.exists():
            raise FileNotFoundError(
                f"BMPD ROI directory does not exist: {roi_dir}"
            )
    
        if not roi_dir.is_dir():
            raise NotADirectoryError(
                f"BMPD ROI path is not a directory: {roi_dir}"
            )
    
        return sorted(
            path
            for path in roi_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )


    def collect_samples(self) -> list[BMPDSample]:
        """
        Collect BMPD image samples and assign identity labels.

        Returns
        -------
        list of BMPDSample
            Collected BMPD samples with image paths, labels, identity IDs,
            person IDs, and hand sides.

        Raises
        ------
        FileNotFoundError
            If the configured root directory does not exist.
        NotADirectoryError
            If the configured root path is not a directory.
        ValueError
            If one or more filenames cannot be parsed.
        """
        image_paths = self._collect_image_paths()

        if self.config.quality_filter.apply_quality_filter:
            image_paths = self._filter_bad_rois(image_paths)

        identity_to_paths: dict[str, list[Path]] = defaultdict(list)

        for image_path in image_paths:
            person_id, hand = self.parse_bmpd_filename(image_path)
            identity_id = f"{person_id}_{hand}"
            identity_to_paths[identity_id].append(image_path)

        identity_ids = sorted(identity_to_paths)
        identity_to_label = {
            identity_id: label for label, identity_id in enumerate(identity_ids)
        }

        samples: list[BMPDSample] = []

        for identity_id in identity_ids:
            person_id, hand = identity_id.split("_")
            label = identity_to_label[identity_id]

            for image_path in identity_to_paths[identity_id]:
                samples.append(
                    BMPDSample(
                        img_path=str(image_path),
                        label=label,
                        identity_id=identity_id,
                        person_id=person_id,
                        hand=hand,
                    )
                )

        return samples
    
    
    def build_splits(self) -> dict[str, list[BMPDSample]]:
        """
        Build train, validation, and test splits for BMPD.

        Returns
        -------
        dict of str to list of BMPDSample
            Dictionary containing ``"train"``, ``"val"``, and ``"test"``
            sample lists.

        Notes
        -----
        Splitting is performed by identity, not by individual image. This keeps
        all images for the same person-hand identity in the same split and
        helps prevent data leakage.
        """
        samples = self.collect_samples()

        identity_ids = sorted({sample.identity_id for sample in samples})

        split_ids = self.split_identity_ids(
            identity_ids=identity_ids,
            train_ratio=self.config.split.train_ratio,
            val_ratio=self.config.split.val_ratio,
            test_ratio=self.config.split.test_ratio,
            seed=self.config.split.seed,
        )

        samples_by_identity: dict[str, list[BMPDSample]] = defaultdict(list)

        # Group samples so each identity can be added to a split as a unit.
        for sample in samples:
            samples_by_identity[sample.identity_id].append(sample)

        splits: dict[str, list[BMPDSample]] = {}

        for split_name, ids in split_ids.items():
            split_samples: list[BMPDSample] = []

            for identity_id in ids:
                split_samples.extend(samples_by_identity[identity_id])

            splits[split_name] = split_samples

        return splits


    def _filter_bad_rois(self, image_paths: list[Path]) -> list[Path]:
        """
        Filter known bad ROI image paths using the configured quality CSV.

        Parameters
        ----------
        image_paths : list of pathlib.Path
            Candidate image paths before quality filtering.

        Returns
        -------
        list of pathlib.Path
            Image paths remaining after quality filtering.
        """
        project_root = self._get_project_root()
        quality_csv_path = self._get_quality_csv_path(project_root)

        filtered_paths, _qc_stats = filter_bad_roi_paths(
            image_paths=image_paths,
            quality_csv_path=quality_csv_path,
            project_root=project_root,
            verbose=self.config.quality_filter.verbose,
        )

        return filtered_paths


    def _get_project_root(self) -> Path:
        """
        Return the configured or inferred project root.

        Returns
        -------
        pathlib.Path
            Project root directory.

        Raises
        ------
        ValueError
            If the project root cannot be inferred.
        """
        if self.config.quality_filter.project_root is not None:
            return self.config.quality_filter.project_root.resolve()

        root_dir = self.config.roi_dir.resolve()

        try:
            return root_dir.parents[2]
        except IndexError as exc:
            raise ValueError(
                "Could not infer project_root from root_dir. "
                "Pass project_root explicitly in BMPDDatasetConfig."
            ) from exc


    def _get_quality_csv_path(self, project_root: Path) -> Path:
        """
        Return the configured or inferred ROI quality CSV path.

        Parameters
        ----------
        project_root : pathlib.Path
            Project root directory.

        Returns
        -------
        pathlib.Path
            Path to the ROI quality CSV.
        """
        if self.config.quality_filter.quality_csv_path is not None:
            return self.config.quality_filter.quality_csv_path.resolve()

        return project_root / "data" / "qc" / "roi_quality.csv"


    @staticmethod
    def split_identity_ids(
        identity_ids: Iterable[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> dict[str, list[str]]:
        """
        Split identity IDs into train, validation, and test sets.

        Parameters
        ----------
        identity_ids : iterable of str
            Identity IDs to split, such as ``"023_L"``.
        train_ratio : float, default=0.7
            Proportion of identities assigned to training.
        val_ratio : float, default=0.1
            Proportion of identities assigned to validation.
        test_ratio : float, default=0.2
            Proportion of identities assigned to testing.
        seed : int, default=42
            Random seed used to shuffle identities before splitting.

        Returns
        -------
        dict of str to list of str
            Dictionary with keys ``"train"``, ``"val"``, and ``"test"``.

        Raises
        ------
        ValueError
            If ratios are invalid or no identities are provided.
        """
        BMPDDatasetBuilder._validate_split_ratios(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        shuffled_ids = list(identity_ids)

        if not shuffled_ids:
            raise ValueError("At least one identity ID is required for splitting.")

        random.Random(seed).shuffle(shuffled_ids)

        n_total = len(shuffled_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train : n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val :]

        return {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }


    @staticmethod
    def _validate_split_ratios(
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Validate train, validation, and test split ratios.

        Parameters
        ----------
        train_ratio : float
            Training split ratio.
        val_ratio : float
            Validation split ratio.
        test_ratio : float
            Test split ratio.
        tolerance : float, default=1e-6
            Allowed floating-point tolerance for checking the ratio sum.

        Raises
        ------
        ValueError
            If any ratio is negative or if the ratios do not sum to 1.0.
        """
        ratios = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }

        for name, value in ratios.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative. Got {value}.")

        ratio_sum = train_ratio + val_ratio + test_ratio

        if abs(ratio_sum - 1.0) > tolerance:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0. "
                f"Got {ratio_sum:.6f}."
            )

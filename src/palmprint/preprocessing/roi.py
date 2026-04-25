"""
Utilities for palmprint ROI extraction and hand segmentation.

This module provides reusable, image-level preprocessing functions for deriving
candidate palm ROIs from raw hand images. Supported operations include
grayscale conversion, illumination normalization, threshold-based mask
generation, morphological cleanup, connected-component filtering, contour
extraction, bounding-box cropping, and center- or centroid-based ROI cropping.

The ROI pipeline is still experimental, so the functions in this module are
designed to be configurable, composable, and easy to test in notebooks,
dataset-preparation scripts, and training workflows.
"""
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

# Type aliases
GrayImage = npt.NDArray[np.uint8]
BgrImage = npt.NDArray[np.uint8]
BinaryMask = npt.NDArray[np.uint8]
Contour = npt.NDArray[np.int32]

ThresholdMethod = Literal["otsu", "adaptive"]
CropMethod = Literal["center", "centroid"]


@dataclass(slots=True)
class CropResult:
    """
    Container for a crop and its bounding box coordinates.

    Attributes
    ----------
    image : np.ndarray
        Cropped image region.
    x0, y0, x1, y1 : int
        Bounding coordinates in the source image.
    """

    image: npt.NDArray[np.uint8]
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(slots=True)
class RoiExtractionConfig:
    """
    Configuration for hand segmentation and ROI extraction.

    Attributes
    ----------
    use_clahe : bool
        Whether to apply CLAHE illumination normalization before thresholding.
    clahe_clip_limit : float
        Contrast limit for CLAHE.
    clahe_tile_grid_size : tuple[int, int]
        Tile size for CLAHE histogram equalization.
    blur_ksize : tuple[int, int]
        Gaussian blur kernel size. OpenCV expects odd, positive dimensions.
    threshold_method : str
        Method used to compute the binary mask threshold.
        "otsu" uses global Otsu thresholding.
        "adaptive" uses local adaptive thresholding.
    adaptive_block_size : int
        Neighborhood size for adaptive thresholding. Must be odd and > 1.
    adaptive_c : float
        Constant subtracted from the local mean in adaptive thresholding.
    adaptive_invert : bool
        If True, invert the adaptive threshold mask
    morph_kernel_size : tuple[int, int]
        Kernel size for morphological cleanup.
    morph_close_iterations : int
        Number of closing iterations to fill gaps / holes.
    morph_open_iterations : int
        Number of opening iterations to remove small noise.
    morph_close_first : bool
        If True, apply morphological closing then opening.
    bbox_margin_frac : float
        Extra padding around the contour bounding box as a fraction of its size.
    roi_crop_frac : float
        Fraction of the tight crop width/height retained for the centroid crop.
    crop_method : str
        Strategy used to crop the ROI.
        "center" crops around the image center.
        "centroid" crops around the mask foreground centroid.
    """

    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    blur_ksize: tuple[int, int] = (7, 7)

    threshold_method: ThresholdMethod = "otsu"  # "otsu" or "adaptive"
    adaptive_block_size: int = 31
    adaptive_c: float = 5.0
    adaptive_invert: bool = True

    morph_kernel_size: tuple[int, int] = (11, 11)
    morph_close_iterations: int = 2
    morph_open_iterations: int = 1
    morph_close_first: bool = True

    bbox_margin_frac: float = 0.03
    roi_crop_frac: float = 0.65
    crop_method: CropMethod = "centroid"   # "center" or "centroid"
    
    rotate_to_principal_axis: bool = False
    RotationCenter = Literal["centroid", "image_center"]
    rotation_center: RotationCenter = "centroid"
    
    valley_roi_size: int = 224
    valley_crop_width_scale: float = 2.2
    valley_crop_height_scale: float = 2.2
    valley_down_scale: float = 1.15
    min_defect_depth: float = 20.0


@dataclass(slots=True)
class RoiExtractionResult:
    """
    Intermediate and final outputs from the ROI extraction pipeline.

    Attributes
    ----------
    gray : np.ndarray
        Grayscale version of the input image.
    normalized : np.ndarray
        Intensity-normalized grayscale image used for thresholding.
    blurred : np.ndarray
        Smoothed image produced by Gaussian or similar blur to reduce noise.
    raw_mask : np.ndarray
        Initial binary mask obtained from thresholding.
    cleaned_mask : np.ndarray
        Binary mask after morphological cleanup (e.g., close/open).
    hand_mask : np.ndarray
        Mask of the largest connected component representing the hand.
    tight_crop_bgr : CropResult
        Color image tightly cropped around the detected hand contour.
        Coordinates reference the original input image.
    tight_crop_mask : CropResult
        Binary hand mask cropped to the same bounding box as the tight crop.
        Coordinates reference the original input image.
    final_roi : CropResult
        Final region-of-interest crop extracted from the tight hand crop.
        Coordinates reference the tight crop image.
    """

    gray: GrayImage
    normalized: GrayImage
    blurred: GrayImage
    raw_mask: BinaryMask
    cleaned_mask: BinaryMask
    hand_mask: BinaryMask
    tight_crop_bgr: CropResult
    tight_crop_mask: CropResult
    final_roi: CropResult
    rotation_angle_deg: float | None = None
    rotated_bgr: BgrImage | None = None
    rotated_mask: BinaryMask | None = None
    rotated_tight_crop_bgr: BgrImage | None = None
    rotated_tight_crop_mask: BinaryMask | None = None


def to_grayscale(image: BgrImage) -> GrayImage:
    """
    Convert a BGR image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image in OpenCV BGR format with shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Grayscale image with shape (H, W).

    Notes
    -----
    This is a linear color-space conversion performed by OpenCV.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    gray: GrayImage,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> GrayImage:
    """
    Normalize local contrast using CLAHE.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 grayscale image.
    clip_limit : float, optional
        CLAHE contrast clip limit.
    tile_grid_size : tuple[int, int], optional
        Tile size used for local histogram equalization.

    Returns
    -------
    np.ndarray
        Contrast-normalized grayscale image.

    Notes
    -----
    CLAHE can improve thresholding robustness under uneven illumination,
    which is common in mobile palmprint images like BMPD.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(gray)


def gaussian_blur(
    image: GrayImage,
    *,
    ksize: tuple[int, int] = (7, 7),
    sigma_x: float = 0.0,
) -> GrayImage:
    """
    Apply Gaussian smoothing.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    ksize : tuple[int, int], optional
        Odd-valued Gaussian kernel size.
    sigma_x : float, optional
        Standard deviation in the x. A value of 0 lets OpenCV infer it.

    Returns
    -------
    np.ndarray
        Blurred image.

    Notes
    -----
    Gaussian smoothing reduces high-frequency noise before thresholding.
    """
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x)


def compute_otsu_mask(gray: GrayImage) -> tuple[BinaryMask, float]:
    """
    Compute a binary mask using Otsu's thresholding.

    Parameters
    ----------
    gray : np.ndarray
        Input grayscale image.

    Returns
    -------
    mask : np.ndarray
        Binary mask with values in {0, 255}.
    threshold : float
        Otsu-selected global threshold value.

    Notes
    -----
    Otsu's method minimizes variance between foreground and background classes.
    It works best when the histogram is reasonably bimodal.
    """
    threshold, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return mask, float(threshold)


def compute_adaptive_mask(
    gray: GrayImage,
    *,
    block_size: int = 31,
    c: float = 5.0,
    invert: bool = True,
) -> BinaryMask:
    """
    Compute a binary mask using adaptive Gaussian thresholding.

    Parameters
    ----------
    gray : np.ndarray
        Input grayscale image.
    block_size : int, optional
        Odd neighborhood size used to compute local thresholds.
    c : float, optional
        Constant subtracted from the local weighted mean.
    invert : bool, optional
        Whether to return an inverted binary mask.

    Returns
    -------
    np.ndarray
        Binary mask with values in {0, 255}.

    Notes
    -----
    Adaptive thresholding is often more robust than Otsu under strong lighting
    gradients, but it can also introduce background artifacts if parameters are
    poor.
    """
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        c,
    )


def clean_binary_mask(
    mask: BinaryMask,
    *,
    kernel_size: tuple[int, int] = (11, 11),
    close_iterations: int = 2,
    open_iterations: int = 1,
    close_first: bool = True,
) -> BinaryMask:
    """
    Clean a binary mask with morphological closing and opening.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask with values in {0, 255}.
    kernel_size : tuple[int, int], optional
        Structuring element size.
    close_iterations : int, optional
        Number of closing iterations. Helps fill small holes and connect gaps.
    open_iterations : int, optional
        Number of opening iterations. Helps remove small foreground noise.
    close_first : bool, optional
        Whether to apply closing before opening

    Returns
    -------
    np.ndarray
        Cleaned binary mask.

    Notes
    -----
    Closing = dilation followed by erosion.
    Opening = erosion followed by dilation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    cleaned = mask.copy()

    if close_first:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations
        )
    else:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations
        )
    return cleaned


def extract_largest_component(mask: BinaryMask) -> BinaryMask:
    """
    Keep only the largest connected foreground component.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask with foreground as nonzero pixels.

    Returns
    -------
    np.ndarray
        Binary mask containing only the largest connected component.

    Notes
    -----
    Connected-component labeling treats each contiguous foreground region as
    a separate object. This is useful when thresholding creates unwanted blobs.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    # If no foreground component exists, return a copy.
    if num_labels <= 1:
        return mask.copy()

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.zeros_like(mask)
    component[labels == largest_label] = 255
    return component


def extract_largest_contour(mask: BinaryMask) -> Contour:
    """
    Extract the largest external contour from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary foreground mask.

    Returns
    -------
    np.ndarray
        Largest contour as an OpenCV contour array.

    Raises
    ------
    RuntimeError
        If no contours are found.

    Notes
    -----
    The largest contour is often a good approximation of the hand boundary
    after connected-component cleanup.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "No contours found. Check mask polarity or preprocessing settings."
        )
    return max(contours, key=cv2.contourArea)


def crop_to_bounding_box(
    image: npt.NDArray[np.uint8],
    contour: Contour,
    *,
    margin_frac: float = 0.03,
) -> CropResult:
    """
    Crop an image to the contour's bounding box with optional margin.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale or color.
    contour : np.ndarray
        Contour defining the object boundary.
    margin_frac : float, optional
        Margin added to each side as a fraction of bbox width/height.

    Returns
    -------
    CropResult
        Cropped image and source coordinates.

    Raises
    ------
    ValueError
        If ``margin_frac`` is negative.
    """
    if margin_frac < 0:
        raise ValueError("margin_frac must be non-negative.")

    x, y, w, h = cv2.boundingRect(contour)
    mx = int(w * margin_frac)
    my = int(h * margin_frac)

    height, width = image.shape[:2]

    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(width, x + w + mx)
    y1 = min(height, y + h + my)

    cropped = image[y0:y1, x0:x1]
    return CropResult(image=cropped, x0=x0, y0=y0, x1=x1, y1=y1)


def center_crop_image(
    image: npt.NDArray[np.uint8],
    *,
    crop_frac: float = 0.65,
) -> CropResult:
    """
    Crop the central region of an image.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color image.
    crop_frac : float, optional
        Fraction of width/height to retain.

    Returns
    -------
    CropResult
        Center crop and coordinates.
    """
    if not 0 < crop_frac <= 1:
        raise ValueError("crop_frac must be in the interval (0, 1].")
    height, width = image.shape[:2]
    crop_w = int(width * crop_frac)
    crop_h = int(height * crop_frac)

    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    cropped = image[y0:y1, x0:x1]
    return CropResult(image=cropped, x0=x0, y0=y0, x1=x1, y1=y1)


def crop_around_mask_centroid(
    image: npt.NDArray[np.uint8],
    mask: BinaryMask,
    *,
    crop_frac: float = 0.65,
) -> CropResult:
    """
    Crop around the centroid of foreground pixels in a mask.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color image.
    mask : np.ndarray
        Binary mask aligned with `image`.
    crop_frac : float, optional
        Fraction of image width/height to retain.

    Returns
    -------
    CropResult
        Centroid-centered crop and coordinates.

    Raises
    ------
    ValueError
        If ``crop_frac`` is not in the interval ``(0, 1]``.
    RuntimeError
        If the mask contains no foreground pixels.

    Notes
    -----
    The centroid is computed as the mean of foreground pixel coordinates:
        cx = mean(x_i), cy = mean(y_i)
    This is a discrete approximation of the mask's center of mass.
    """
    if not 0 < crop_frac <= 1:
        raise ValueError("crop_frac must be in the interval (0, 1].")

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise RuntimeError("No foreground pixels found in mask.")

    cx = int(xs.mean())
    cy = int(ys.mean())

    height, width = image.shape[:2]
    crop_w = int(width * crop_frac)
    crop_h = int(height * crop_frac)

    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(width, x0 + crop_w)
    y1 = min(height, y0 + crop_h)

    # If the crop clipped at the right or bottom boundary, shift it back so the
    # output size remains as close as possible to the requested crop dimensions.
    x0 = max(0, x1 - crop_w)
    y0 = max(0, y1 - crop_h)

    cropped = image[y0:y1, x0:x1]
    return CropResult(image=cropped, x0=x0, y0=y0, x1=x1, y1=y1)


@dataclass(slots=True)
class PrincipalAxisResult:
    """
    PCA-based orientation information derived from a binary hand mask.

    Attributes
    ----------
    angle_deg : float
        Angle of the first principal axis in degrees, measured from the
        positive x-axis in image coordinates.
    centroid_xy : tuple[float, float]
        Mean x/y coordinate of the foreground mask pixels.
    eigenvectors : np.ndarray
        Principal directions returned by OpenCV PCA.
    eigenvalues : np.ndarray
        Variances along the principal directions.
    """
    angle_deg: float
    centroid_xy: tuple[float, float]
    eigenvectors: npt.NDArray[np.float32]
    eigenvalues: npt.NDArray[np.float32]


@dataclass(slots=True)
class RotationResult:
    """
    Result of an affine in-place image rotation.

    Attributes
    ----------
    image : np.ndarray
        Rotated output image with the same spatial dimensions as the input.
    matrix : np.ndarray
        2x3 affine transformation matrix returned by OpenCV.
    """
    image: npt.NDArray[np.uint8]
    matrix: npt.NDArray[np.float64]
    

def rotate_image(
    image: npt.NDArray[np.uint8],
    angle_deg: float,
    *,
    center: tuple[float, float] | None = None,
    interp: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | tuple[int, int, int] = 0,
) -> RotationResult:
    """
    Rotate an image in place while preserving its original output size.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or color image.
    angle_deg : float
        Rotation angle in degrees. Positive values follow OpenCV's
        counterclockwise convention.
    center : tuple[float, float] | None, optional
        Rotation center as (x, y). If None, the image center is used.
    interp : int, optional
        OpenCV interpolation flag. Use INTER_LINEAR for natural images and
        INTER_NEAREST for masks.
    border_mode : int, optional
        OpenCV border mode used outside the image bounds.
    border_value : int | tuple[int, int, int], optional
        Fill value when border_mode is BORDER_CONSTANT.

    Returns
    -------
    RotationResult
        Rotated image and the affine transform matrix.
    """
    height, width = image.shape[:2]

    if center is None:
        center = ((width - 1) / 2.0, (height - 1) / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interp,
        borderMode=border_mode,
        borderValue=border_value,
    )
    return RotationResult(image=rotated, matrix=matrix)


def compute_mask_principal_axis(mask: BinaryMask) -> PrincipalAxisResult:
    """
    Compute the dominant orientation of a binary mask using PCA on foreground
    pixel coordinates.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask with foreground pixels > 0.

    Returns
    -------
    PrincipalAxisResult
        PCA orientation information for the foreground mask.

    Raises
    ------
    RuntimeError
        If the mask contains fewer than two foreground pixels.
    """
    ys, xs = np.where(mask > 0)

    if len(xs) < 2:
        raise RuntimeError("Not enough foreground pixels for PCA.")

    pts = np.column_stack((xs, ys)).astype(np.float32)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

    vx, vy = eigenvectors[0]
    angle_rad = np.arctan2(vy, vx)
    angle_deg = float(np.degrees(angle_rad))

    centroid_xy = (float(mean[0, 0]), float(mean[0, 1]))

    return PrincipalAxisResult(
        angle_deg=angle_deg,
        centroid_xy=centroid_xy,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
    )


def extract_hand_roi(
    image: BgrImage,
    config: RoiExtractionConfig,
) -> RoiExtractionResult:
    """
    Extract a hand region of interest (ROI) from a color image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image as an unsigned 8-bit NumPy array.
    config : RoiExtractionConfig
        Configuration controlling normalization, thresholding,
        morphological cleanup, and final ROI cropping.

    Returns
    -------
    RoiExtractionResult
        Dataclass containing intermediate pipeline outputs and the final
        extracted ROI.

    Raises
    ------
    ValueError
        If ``config.threshold_method`` is unsupported.
    ValueError
        If ``config.crop_method`` is unsupported.
    """
    # Convert the input image to grayscale for preprocessing
    gray = to_grayscale(image)

    # Optionally normalize local contrast
    if config.use_clahe:
        normalized = apply_clahe(
            gray,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )
    else:
        normalized = gray.copy()

    # Smooth the normalized image to reduce noise before thresholding
    blurred = gaussian_blur(
        normalized,
        ksize=config.blur_ksize,
    )

    # Compute the initial binary foreground mask
    if config.threshold_method == "otsu":
        raw_mask, _ = compute_otsu_mask(blurred)
    elif config.threshold_method == "adaptive":
        raw_mask = compute_adaptive_mask(
            blurred,
            block_size=config.adaptive_block_size,
            c=config.adaptive_c,
            invert=config.adaptive_invert,
        )
    else:
        raise ValueError(
            f"Unsupported threshold method: {config.threshold_method}"
        )

    # Clean raw mask with morphology
    cleaned_mask = clean_binary_mask(
        raw_mask,
        kernel_size=config.morph_kernel_size,
        close_iterations=config.morph_close_iterations,
        open_iterations=config.morph_open_iterations,
        close_first=config.morph_close_first,
    )

    # Keep only the largest connected component
    hand_mask = extract_largest_component(cleaned_mask)
    hand_contour = extract_largest_contour(hand_mask)

    # Crop a tight region around the detected hand from both original image
    # and the hand mask
    tight_crop_bgr_result = crop_to_bounding_box(
        image,
        hand_contour,
        margin_frac=config.bbox_margin_frac,
    )
    tight_crop_mask_result = crop_to_bounding_box(
        hand_mask,
        hand_contour,
        margin_frac=config.bbox_margin_frac,
    )

    tight_crop_bgr = tight_crop_bgr_result.image
    tight_crop_mask = tight_crop_mask_result.image
    
    rotation_angle_deg = None
    rotated_bgr = None
    rotated_mask = None
    rotated_tight_crop_bgr = None
    rotated_tight_crop_mask = None
    
    crop_source_bgr = tight_crop_bgr
    crop_source_mask = tight_crop_mask
    
    if config.rotate_to_principal_axis:
        axis_result = compute_mask_principal_axis(tight_crop_mask)
        rotation_angle_deg = -axis_result.angle_deg
    
        if config.rotation_center == "centroid":
            center = axis_result.centroid_xy
        elif config.rotation_center == "image_center":
            h, w = tight_crop_mask.shape[:2]
            center = ((w - 1) / 2.0, (h - 1) / 2.0)
        else:
            raise ValueError(f"Unsupported rotation_center: {config.rotation_center}")
    
        rotated_bgr_result = rotate_image(
            tight_crop_bgr,
            rotation_angle_deg,
            center=center,
            interp=cv2.INTER_LINEAR,
            border_value=0,
        )
        rotated_mask_result = rotate_image(
            tight_crop_mask,
            rotation_angle_deg,
            center=center,
            interp=cv2.INTER_NEAREST,
            border_value=0,
        )
    
        rotated_bgr = rotated_bgr_result.image
        rotated_mask = extract_largest_component(rotated_mask_result.image)
    
        rotated_contour = extract_largest_contour(rotated_mask)
    
        rotated_tight_crop_bgr = crop_to_bounding_box(
            rotated_bgr,
            rotated_contour,
            margin_frac=config.bbox_margin_frac,
        ).image
        rotated_tight_crop_mask = crop_to_bounding_box(
            rotated_mask,
            rotated_contour,
            margin_frac=config.bbox_margin_frac,
        ).image
    
        crop_source_bgr = rotated_tight_crop_bgr
        crop_source_mask = rotated_tight_crop_mask

    # Extract the final ROI from the tight crop using configured cropping
    # strategy
    if config.crop_method == "center":
        final_roi_result = center_crop_image(
            crop_source_bgr,
            crop_frac=config.roi_crop_frac,
        )
    elif config.crop_method == "centroid":
        final_roi_result = crop_around_mask_centroid(
            crop_source_bgr,
            crop_source_mask,
            crop_frac=config.roi_crop_frac,
        )
    else:
        raise ValueError(f"Unsupported crop method: {config.crop_method}")

    return RoiExtractionResult(
        gray=gray,
        normalized=normalized,
        blurred=blurred,
        raw_mask=raw_mask,
        cleaned_mask=cleaned_mask,
        hand_mask=hand_mask,
        tight_crop_bgr=tight_crop_bgr_result,
        tight_crop_mask=tight_crop_mask_result,
        final_roi=final_roi_result,
        rotation_angle_deg=rotation_angle_deg,
        rotated_bgr=rotated_bgr,
        rotated_mask=rotated_mask,
        rotated_tight_crop_bgr=rotated_tight_crop_bgr,
        rotated_tight_crop_mask=rotated_tight_crop_mask,
    )


@dataclass(slots=True)
class ValleyLandmarkResult:
    valley_points: list[tuple[int, int]]
    selected_valleys: tuple[tuple[int, int], tuple[int, int]]
    midpoint_xy: tuple[float, float]
    angle_deg: float
    valley_distance: float


@dataclass(slots=True)
class ValleyRoiExtractionResult:
    base_result: RoiExtractionResult
    landmarks: ValleyLandmarkResult
    aligned_bgr: BgrImage
    aligned_mask: BinaryMask
    final_roi: CropResult
    

def find_valley_points_from_defects(
    contour: Contour,
    *,
    min_defect_depth: float = 20.0,
) -> list[tuple[int, int]]:
    hull = cv2.convexHull(contour, returnPoints=False)

    if hull is None or len(hull) < 4:
        return []

    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return []

    valleys = []

    for defect in defects[:, 0]:
        _start_idx, _end_idx, far_idx, depth = defect

        # OpenCV stores depth scaled by 256
        depth_px = depth / 256.0

        if depth_px < min_defect_depth:
            continue

        far_point = contour[far_idx][0]
        valleys.append((int(far_point[0]), int(far_point[1])))

    return valleys


def select_two_upper_valleys(
    valley_points: list[tuple[int, int]],
    mask: BinaryMask,
    *,
    upper_frac: float = 0.55,
    max_pair_angle_deg: float = 40.0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Select two likely finger-valley landmarks from convexity-defect points.

    This filters out lower wrist/palm defects, rejects highly diagonal pairs,
    then selects the middle pair from the remaining left-to-right valley points.
    """
    if len(valley_points) < 2:
        raise RuntimeError("Not enough valley points found for Option C ROI.")

    ys, xs = np.where(mask > 0)
    hand_top = int(ys.min())
    hand_bottom = int(ys.max())
    hand_height = hand_bottom - hand_top

    # 1. Keep only valleys in upper 55% of the hand.
    max_y = hand_top + upper_frac * hand_height
    candidates = [p for p in valley_points if p[1] <= max_y]

    if len(candidates) < 2:
        candidates = valley_points

    # 2. Sort left-to-right.
    candidates = sorted(candidates, key=lambda p: p[0])

    # 3. Build near-horizontal neighbor pairs only.
    valid_pairs = []

    for i in range(len(candidates) - 1):
        p1 = candidates[i]
        p2 = candidates[i + 1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if dx == 0:
            continue

        angle_deg = abs(float(np.degrees(np.arctan2(dy, dx))))

        if angle_deg <= max_pair_angle_deg:
            valid_pairs.append((p1, p2))

    # 4. Choose the middle valid pair instead of the widest pair.
    if valid_pairs:
        return valid_pairs[len(valid_pairs) // 2]

    # Fallback: choose middle pair from all upper candidates.
    if len(candidates) >= 2:
        mid = len(candidates) // 2
        return candidates[mid - 1], candidates[mid]

    raise RuntimeError("Could not select two valley points.")


def extract_valley_landmark_roi(
    image: BgrImage,
    config: RoiExtractionConfig,
) -> ValleyRoiExtractionResult:
    base_result = extract_hand_roi(image, config)

    hand_mask = base_result.hand_mask
    contour = extract_largest_contour(hand_mask)

    valley_points = find_valley_points_from_defects(
        contour,
        min_defect_depth=config.min_defect_depth,
    )

    v1, v2 = select_two_upper_valleys(
        valley_points,
        hand_mask,
        upper_frac=0.55,
        max_pair_angle_deg=40.0,
    )

    p1 = np.array(v1, dtype=np.float32)
    p2 = np.array(v2, dtype=np.float32)

    midpoint = (p1 + p2) / 2.0
    dx, dy = p2 - p1

    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    valley_distance = float(np.linalg.norm(p2 - p1))

    # Rotate so the valley-to-valley line is horizontal.
    rotate_by = -angle_deg

    aligned_bgr_result = rotate_image(
        image,
        rotate_by,
        center=(float(midpoint[0]), float(midpoint[1])),
        interp=cv2.INTER_LINEAR,
        border_value=0,
    )

    aligned_mask_result = rotate_image(
        hand_mask,
        rotate_by,
        center=(float(midpoint[0]), float(midpoint[1])),
        interp=cv2.INTER_NEAREST,
        border_value=0,
    )

    aligned_bgr = aligned_bgr_result.image
    aligned_mask = extract_largest_component(aligned_mask_result.image)

    # Transform valley midpoint into rotated coordinates.
    M = aligned_bgr_result.matrix
    midpoint_h = np.array([midpoint[0], midpoint[1], 1.0], dtype=np.float32)
    aligned_midpoint = M @ midpoint_h

    cx = int(aligned_midpoint[0])
    cy = int(aligned_midpoint[1] + config.valley_down_scale * valley_distance)

    crop_w = int(config.valley_crop_width_scale * valley_distance)
    crop_h = int(config.valley_crop_height_scale * valley_distance)

    h, w = aligned_bgr.shape[:2]

    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(w, x0 + crop_w)
    y1 = min(h, y0 + crop_h)

    x0 = max(0, x1 - crop_w)
    y0 = max(0, y1 - crop_h)

    roi = aligned_bgr[y0:y1, x0:x1]

    if config.valley_roi_size is not None:
        roi = cv2.resize(
            roi,
            (config.valley_roi_size, config.valley_roi_size),
            interpolation=cv2.INTER_AREA,
        )

    final_roi = CropResult(
        image=roi,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )

    landmarks = ValleyLandmarkResult(
        valley_points=valley_points,
        selected_valleys=(v1, v2),
        midpoint_xy=(float(midpoint[0]), float(midpoint[1])),
        angle_deg=angle_deg,
        valley_distance=valley_distance,
    )

    return ValleyRoiExtractionResult(
        base_result=base_result,
        landmarks=landmarks,
        aligned_bgr=aligned_bgr,
        aligned_mask=aligned_mask,
        final_roi=final_roi,
    )

@dataclass(slots=True)
class TopBoundaryValleyRoiResult:
    base_result: RoiExtractionResult
    boundary_points: list[tuple[int, int]]
    valley_points: list[tuple[int, int]]
    selected_valleys: tuple[tuple[int, int], tuple[int, int]]
    aligned_bgr: BgrImage
    aligned_mask: BinaryMask
    final_roi: CropResult
    angle_deg: float
    valley_distance: float
    

def smooth_1d(values: np.ndarray, window_size: int = 31) -> np.ndarray:
    """
    Smooth a 1D signal using a moving average.
    """
    if window_size < 3:
        return values.copy()

    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(values, kernel, mode="same")


def extract_top_boundary_curve(
    mask: BinaryMask,
    *,
    upper_frac: float = 0.65,
) -> list[tuple[int, int]]:
    """
    For each x-column, find the first foreground pixel from the top.
    This traces the upper boundary of the segmented hand.
    """
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        raise RuntimeError("No foreground pixels found in mask.")

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    max_y = int(y_min + upper_frac * (y_max - y_min))

    boundary_points = []

    for x in range(x_min, x_max + 1):
        column_ys = np.where(mask[:, x] > 0)[0]

        if column_ys.size == 0:
            continue

        y = int(column_ys.min())

        if y <= max_y:
            boundary_points.append((x, y))

    return boundary_points


def find_top_boundary_valleys(
    boundary_points: list[tuple[int, int]],
    *,
    smooth_window: int = 41,
    min_prominence: float = 20.0,
    min_spacing_frac: float = 0.08,
) -> list[tuple[int, int]]:
    """
    Find downward dips in the top-boundary curve.

    Important image-coordinate detail:
    - Smaller y = higher in the image
    - Larger y = lower in the image
    So finger valleys are local maxima in y.
    """
    if len(boundary_points) < smooth_window:
        return []

    xs = np.array([p[0] for p in boundary_points], dtype=np.int32)
    ys = np.array([p[1] for p in boundary_points], dtype=np.float32)

    ys_smooth = smooth_1d(ys, window_size=smooth_window)

    x_range = xs.max() - xs.min()
    min_spacing = max(10, int(min_spacing_frac * x_range))

    candidate_indices = []

    for i in range(1, len(ys_smooth) - 1):
        left = ys_smooth[i - 1]
        center = ys_smooth[i]
        right = ys_smooth[i + 1]

        # Local maximum in y means a downward dip in image coordinates.
        if center > left and center > right:
            candidate_indices.append(i)

    if not candidate_indices:
        return []

    # Estimate prominence by comparing point to nearby neighborhood.
    valleys = []

    for i in candidate_indices:
        left_start = max(0, i - min_spacing)
        right_end = min(len(ys_smooth), i + min_spacing + 1)

        local_min = min(
            float(ys_smooth[left_start:i].min()) if i > left_start else ys_smooth[i],
            float(ys_smooth[i + 1:right_end].min()) if i + 1 < right_end else ys_smooth[i],
        )

        prominence = float(ys_smooth[i] - local_min)

        if prominence >= min_prominence:
            valleys.append((int(xs[i]), int(ys_smooth[i])))

    # Remove valleys that are too close together.
    valleys = sorted(valleys, key=lambda p: p[0])
    filtered = []

    for point in valleys:
        if not filtered:
            filtered.append(point)
            continue

        prev = filtered[-1]

        if abs(point[0] - prev[0]) >= min_spacing:
            filtered.append(point)
        elif point[1] > prev[1]:
            # Keep the deeper dip.
            filtered[-1] = point

    return filtered


def select_internal_boundary_valleys(
    valley_points: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Select internal finger valleys, avoiding extreme left/right edge valleys.
    """
    if len(valley_points) < 2:
        raise RuntimeError("Not enough top-boundary valleys found.")

    valleys = sorted(valley_points, key=lambda p: p[0])

    # If we have many valleys, avoid the outermost points because those are
    # often thumb/pinky edges rather than true finger valleys.
    if len(valleys) >= 4:
        internal = valleys[1:-1]
    else:
        internal = valleys

    if len(internal) >= 2:
        mid = len(internal) // 2
        return internal[mid - 1], internal[mid]

    return valleys[0], valleys[-1]


def extract_top_boundary_valley_roi(
    image: BgrImage,
    config: RoiExtractionConfig,
    *,
    rough_rotate_first: bool = True,
    smooth_window: int = 41,
    min_prominence: float = 20.0,
) -> TopBoundaryValleyRoiResult:
    """
    Extract palm ROI using top-boundary valley detection.

    This method:
    1. Segments the hand.
    2. Roughly aligns the hand with PCA.
    3. Finds the upper hand boundary.
    4. Detects downward dips as candidate finger valleys.
    5. Uses two internal valleys to align and crop the palm.
    """
    base_result = extract_hand_roi(image, config)

    source_bgr = image
    source_mask = base_result.hand_mask

    rough_angle = 0.0

    if rough_rotate_first:
        axis_result = compute_mask_principal_axis(source_mask)
        rough_angle = -axis_result.angle_deg

        rough_bgr_result = rotate_image(
            source_bgr,
            rough_angle,
            center=axis_result.centroid_xy,
            interp=cv2.INTER_LINEAR,
            border_value=0,
        )

        rough_mask_result = rotate_image(
            source_mask,
            rough_angle,
            center=axis_result.centroid_xy,
            interp=cv2.INTER_NEAREST,
            border_value=0,
        )

        source_bgr = rough_bgr_result.image
        source_mask = extract_largest_component(rough_mask_result.image)

    boundary_points = extract_top_boundary_curve(
        source_mask,
        upper_frac=0.65,
    )

    valley_points = find_top_boundary_valleys(
        boundary_points,
        smooth_window=smooth_window,
        min_prominence=min_prominence,
    )

    v1, v2 = select_internal_boundary_valleys(valley_points)

    p1 = np.array(v1, dtype=np.float32)
    p2 = np.array(v2, dtype=np.float32)

    midpoint = (p1 + p2) / 2.0
    dx, dy = p2 - p1

    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    valley_distance = float(np.linalg.norm(p2 - p1))

    # Fine alignment: make selected valley line horizontal.
    fine_rotate_by = -angle_deg

    aligned_bgr_result = rotate_image(
        source_bgr,
        fine_rotate_by,
        center=(float(midpoint[0]), float(midpoint[1])),
        interp=cv2.INTER_LINEAR,
        border_value=0,
    )

    aligned_mask_result = rotate_image(
        source_mask,
        fine_rotate_by,
        center=(float(midpoint[0]), float(midpoint[1])),
        interp=cv2.INTER_NEAREST,
        border_value=0,
    )

    aligned_bgr = aligned_bgr_result.image
    aligned_mask = extract_largest_component(aligned_mask_result.image)

    # Transform midpoint after fine rotation.
    M = aligned_bgr_result.matrix
    midpoint_h = np.array([midpoint[0], midpoint[1], 1.0], dtype=np.float32)
    aligned_midpoint = M @ midpoint_h

    cx = int(aligned_midpoint[0])
    cy = int(aligned_midpoint[1] + config.valley_down_scale * valley_distance)

    crop_w = int(config.valley_crop_width_scale * valley_distance)
    crop_h = int(config.valley_crop_height_scale * valley_distance)

    h, w = aligned_bgr.shape[:2]

    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    x1 = min(w, x0 + crop_w)
    y1 = min(h, y0 + crop_h)

    x0 = max(0, x1 - crop_w)
    y0 = max(0, y1 - crop_h)

    roi = aligned_bgr[y0:y1, x0:x1]

    if config.valley_roi_size is not None:
        roi = cv2.resize(
            roi,
            (config.valley_roi_size, config.valley_roi_size),
            interpolation=cv2.INTER_AREA,
        )

    final_roi = CropResult(
        image=roi,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )

    return TopBoundaryValleyRoiResult(
        base_result=base_result,
        boundary_points=boundary_points,
        valley_points=valley_points,
        selected_valleys=(v1, v2),
        aligned_bgr=aligned_bgr,
        aligned_mask=aligned_mask,
        final_roi=final_roi,
        angle_deg=rough_angle + angle_deg,
        valley_distance=valley_distance,
    )
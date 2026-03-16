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
    adaptive_invert: bool = False

    morph_kernel_size: tuple[int, int] = (11, 11)
    morph_close_iterations: int = 2
    morph_open_iterations: int = 1
    morph_close_first: bool = True

    bbox_margin_frac: float = 0.03
    roi_crop_frac: float = 0.65
    crop_method: CropMethod = "centroid"   # "center" or "centroid"


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

    # Extract the final ROI from the tight crop using configured cropping
    # strategy
    if config.crop_method == "center":
        final_roi_result = center_crop_image(
            tight_crop_bgr,
            crop_frac=config.roi_crop_frac,
        )
    elif config.crop_method == "centroid":
        final_roi_result = crop_around_mask_centroid(
            tight_crop_bgr,
            tight_crop_mask,
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
    )

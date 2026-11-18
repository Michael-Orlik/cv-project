"""Prepare image masks and locate text lines."""

import cv2
import numpy as np
from numpy.typing import NDArray

from image_ops import to_grayscale
from preprocessing import binarize_adaptive


def create_foreground_mask(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Create a mask with white foreground and black background.

    Grid lines and image noise may also appear in the foreground.

    Raises:
        ValueError: If the input is not a non-empty RGB uint8 image.
        cv2.error: If OpenCV cannot process the grayscale image.
    """

    # Convert RGB pixels to grayscale for adaptive thresholding.
    grayscale_image = to_grayscale(image)

    # Separate dark regions from the background using local thresholds.
    binary_image = binarize_adaptive(grayscale_image)

    # Mark dark regions as 255 and the background as 0.
    foreground_mask = 255 - binary_image

    return foreground_mask


def detect_grid_lines(
    foreground_mask: NDArray[np.uint8],
    line_length: int = 100,
) -> NDArray[np.uint8]:
    """Find long horizontal and vertical regions in a binary mask.

    The input must be a 2D uint8 mask with white foreground.
    The result may include underlines and parts of letters.

    Raises:
        ValueError: If line_length is not an integer of at least 2.
        cv2.error: If OpenCV cannot process the mask.
    """
    if not isinstance(line_length, int) or line_length < 2:
        raise ValueError("Line length must be an integer of at least 2.")

    # Use a horizontal window to detect long horizontal regions.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (line_length, 1),
    )

    # Use a vertical window to detect long vertical regions.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, line_length),
    )

    horizontal_lines = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    vertical_lines = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    # Include pixels detected in either direction.
    return cv2.bitwise_or(horizontal_lines, vertical_lines)


def remove_grid_lines(
    foreground_mask: NDArray[np.uint8],
    line_length: int = 100,
) -> NDArray[np.uint8]:
    """Remove detected long lines from a foreground mask.

    Some grid fragments may remain; intersecting letter strokes may be removed.

    Raises:
        ValueError: If line_length is not an integer of at least 2.
        cv2.error: If OpenCV cannot process the mask.
    """
    grid_mask = detect_grid_lines(foreground_mask, line_length)

    # Set detected line pixels to 0; keep other foreground pixels.
    cleaned_mask = cv2.subtract(foreground_mask, grid_mask)

    return cleaned_mask


def remove_thin_details(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Suppress thin regions in a binary foreground mask.

    Thin letter strokes and small punctuation marks may also be removed.

    Raises:
        cv2.error: If OpenCV cannot process the mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Remove regions too thin to contain a 3-by-3 window.
    filtered_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return filtered_mask


def calculate_row_density(mask: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Calculate the fraction of foreground pixels in each image row.

    Returns one value per row, ranging from 0 to 1.

    Raises:
        ValueError: If the mask is empty, is not 2D, or is not uint8.
    """
    if mask.ndim != 2 or mask.size == 0 or mask.dtype != np.uint8:
        raise ValueError("Expected a non-empty 2D uint8 mask.")

    # Count foreground pixels separately in each horizontal pixel row.
    foreground_counts = np.count_nonzero(mask, axis=1)

    # Divide by image width to express each count as a fraction.
    row_density = foreground_counts / mask.shape[1]

    return row_density


def smooth_row_density(
    row_density: NDArray[np.float64],
    window_size: int = 15,
) -> NDArray[np.float64]:
    """Average neighboring row densities to reduce short fluctuations.

    Raises:
        ValueError: If the profile is empty or not 1D, or the window size
            is not an odd integer of at least 3.
    """
    if row_density.ndim != 1 or row_density.size == 0:
        raise ValueError("Expected a non-empty 1D density profile.")

    if (
        not isinstance(window_size, int)
        or window_size < 3
        or window_size % 2 == 0
    ):
        raise ValueError("Window size must be an odd integer of at least 3.")

    # Repeat edge values so the averaging window can cover the boundary rows.
    padding = window_size // 2
    padded_density = np.pad(row_density, padding, mode="edge")

    # Give all rows in the averaging window equal weight.
    weights = np.full(window_size, 1.0 / window_size)
    smoothed_density = np.convolve(padded_density, weights, mode="valid")

    return smoothed_density


def find_line_bands(
    smoothed_density: NDArray[np.float64],
    threshold: float = 0.06,
    min_height: int = 20,
) -> list[tuple[int, int]]:
    """Find consecutive rows whose density reaches the threshold.

    Return (top, bottom) pairs, with bottom excluded.
    Bands shorter than min_height pixels are discarded.

    Raises:
        ValueError: If the profile is empty or not 1D, the threshold is
            outside (0, 1], or min_height is not a positive integer
    """
    if smoothed_density.ndim != 1 or smoothed_density.size == 0:
        raise ValueError("Expected a non-empty 1D density profile.")

    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Threshold must be finite and within (0, 1].")

    if not isinstance(min_height, int) or min_height < 1:
        raise ValueError("Minimum height must be a positive integer.")

    bands = []
    start = None

    for row, density in enumerate(smoothed_density):
        if density >= threshold:
            if start is None:
                start = row
        elif start is not None:
            # Close the current band when density drops below the threshold.
            if row - start >= min_height:
                bands.append((start, row))
            start = None

    # Close a band that continues through the final image row.
    if start is not None:
        bottom = smoothed_density.size
        if bottom - start >= min_height:
            bands.append((start, bottom))

    return bands


def expand_line_bands(
    bands: list[tuple[int, int]],
    image_height: int,
    padding: int = 40,
) -> list[tuple[int, int]]:
    """Add vertical padding to candidate bands within the image bounds.

    Expanded bands may overlap and include neighboring text.

    Raises:
        ValueError: If image_height is not a positive integer,
            or padding is not a non-negative integer.
    """
    if not isinstance(image_height, int) or image_height < 1:
        raise ValueError("Image height must be a positive integer.")

    if not isinstance(padding, int) or padding < 0:
        raise ValueError("Padding must be a non-negative integer.")

    expanded_bands = []

    for top, bottom in bands:
        # Keep the expanded coordinates inside the source image.
        expanded_top = max(0, top - padding)
        expanded_bottom = min(image_height, bottom + padding)

        expanded_bands.append((expanded_top, expanded_bottom))

    return expanded_bands


def resolve_band_overlaps(
    bands: list[tuple[int, int]],
    smoothed_density: NDArray[np.float64],
    original_bands: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Split expanded bands in the gap between the original text bands.

    Both band lists must correspond and be ordered from top to bottom.
    The resulting boundaries may still cut letter strokes.
    """
    separated_bands = bands.copy()

    for index in range(len(separated_bands) - 1):
        top, bottom = separated_bands[index]
        next_top, next_bottom = separated_bands[index + 1]

        if bottom > next_top:
            # Keep the separator outside both original text regions.
            search_top = max(next_top, original_bands[index][1])
            search_bottom = min(bottom, original_bands[index + 1][0])
            overlap_density = smoothed_density[search_top:search_bottom]

            # Convert the position within the overlap to an image row.
            boundary = search_top + int(np.argmin(overlap_density))

            separated_bands[index] = (top, boundary)
            separated_bands[index + 1] = (boundary, next_bottom)

    return separated_bands


def find_horizontal_bounds(
    mask: NDArray[np.uint8],
    grayscale: NDArray[np.uint8],
    padding: int = 10,
) -> tuple[int, int]:
    """Estimate text columns from dark regions and add a small margin.

    Return the full width if no suitable regions remain.

    Raises:
        ValueError: If the image arrays or padding are invalid.
    """
    if (
        mask.ndim != 2
        or mask.size == 0
        or mask.dtype != np.uint8
        or grayscale.shape != mask.shape
        or grayscale.dtype != np.uint8
    ):
        raise ValueError("Expected matching non-empty 2D uint8 arrays.")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("Padding must be a non-negative integer.")

    height, width = mask.shape
    brightness_limit, _ = cv2.threshold(
        grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for label, (x, y, w, h, area) in enumerate(stats[1:], start=1):
        if area < max(6, height // 10) or h < 3 or w < 2:
            continue

        # Exclude faint regions, such as writing showing through the paper.
        region = labels[y:y + h, x:x + w] == label
        pixels = grayscale[y:y + h, x:x + w][region]
        if np.median(pixels) <= brightness_limit:
            components.append((int(x), int(x + w), int(area), int(h)))

    if not components:
        return 0, width

    # Group nearby characters while allowing spaces between words.
    max_gap = max(20, int(np.median([part[3] for part in components]) * 2.5))
    groups = []
    for left, right, area, _ in sorted(components):
        if groups and left - groups[-1][1] <= max_gap:
            groups[-1][1] = max(groups[-1][1], right)
            groups[-1][2] += area
        else:
            groups.append([left, right, area])

    largest_area = max(group[2] for group in groups)
    edge_margin = max(2, width // 20)
    text_groups = []
    for left, right, area in groups:
        # Ignore small isolated marks and weak edge groups such as binding holes.
        near_edge = left < edge_margin or right > width - edge_margin
        if area < largest_area * 0.02:
            continue
        if near_edge and area < largest_area * 0.2:
            continue
        text_groups.append((left, right))

    left = max(0, min(group[0] for group in text_groups) - padding)
    right = min(width, max(group[1] for group in text_groups) + padding)
    return left, right


def find_line_boxes(
    image: NDArray[np.uint8],
    threshold: float = 0.06,
    padding: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Find approximate text-line boxes on a single-column page.

    Return (left, top, right, bottom) boxes ordered from top to bottom.
    Right and bottom are excluded. Return an empty list if no bands are found.

    Raises:
        ValueError: If the image or segmentation parameters are invalid.
        cv2.error: If OpenCV cannot process the image masks.
    """
    # Prepare a mask with reduced grid lines and thin details.
    foreground_mask = create_foreground_mask(image)
    cleaned_mask = remove_grid_lines(foreground_mask)
    filtered_mask = remove_thin_details(cleaned_mask)

    # Locate candidate text bands using the horizontal density profile.
    row_density = calculate_row_density(filtered_mask)
    smoothed_density = smooth_row_density(row_density)
    bands = find_line_bands(smoothed_density, threshold=threshold)

    # Expand the bands and separate any overlapping neighbors.
    height = image.shape[0]
    expanded_bands = expand_line_bands(bands, height, padding=padding)
    separated_bands = resolve_band_overlaps(
        expanded_bands,
        smoothed_density,
        original_bands=bands,
    )

    # Estimate columns from each dense text band, excluding neighboring rows.
    grayscale_image = to_grayscale(image)
    boxes = []
    for (band_top, band_bottom), (top, bottom) in zip(bands, separated_bands):
        left, right = find_horizontal_bounds(
            filtered_mask[band_top:band_bottom],
            grayscale_image[band_top:band_bottom],
        )
        boxes.append((left, top, right, bottom))

    return boxes

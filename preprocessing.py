import cv2
import numpy as np
from numpy.typing import NDArray


def binarize(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a uint8 grayscale image to a binary mask using Otsu."""
    # Otsu selects one threshold for the entire image.
    _, binary_image = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return binary_image


def binarize_adaptive(
    image: NDArray[np.uint8],
    block_size: int = 51,
    c: float = 10.0,
) -> NDArray[np.uint8]:
    """Binarize a grayscale image using local Gaussian thresholds."""
    if not isinstance(block_size, int):
        raise ValueError("Block size must be an integer.")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("Block size must be odd and at least 3.")
    if not np.isfinite(c):
        raise ValueError("C must be a finite number.")

    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )

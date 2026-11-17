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

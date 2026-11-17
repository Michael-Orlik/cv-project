import numpy as np
from numpy.typing import NDArray
from PIL import Image

SUPPORTED_NDIMS = (2, 3)


def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert an RGB uint8 image to a separate grayscale array.

    Raises:
        ValueError: If the image is empty, is not uint8, or does not have
            shape (height, width, 3).
    """
    if image.ndim != 3:
        raise ValueError("Expected a three-dimensional image.")
    if image.shape[2] != 3:
        raise ValueError("Expected exactly three color channels.")
    if image.dtype != np.uint8:
        raise ValueError("Expected an image with dtype uint8.")
    if image.size == 0:
        raise ValueError("Expected a non-empty image.")
    # Create a Pillow image from the array.
    pil_image = Image.fromarray(image)
    # Convert the image to grayscale.
    grayscale_image = pil_image.convert("L")
    # Return an independent NumPy array.
    grayscale_array = np.array(grayscale_image, dtype=np.uint8, copy=True)
    return grayscale_array


def crop_image(
    image: NDArray[np.uint8],
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> NDArray[np.uint8]:
    """Copy a rectangular region from a grayscale or RGB image.

    Right and bottom are exclusive. The source image is unchanged.

    Raises:
        ValueError: If the image shape, dtype, or crop bounds are invalid.
        TypeError: If coordinates cannot be used as slice indices.
    """
    if image.ndim not in SUPPORTED_NDIMS:
        raise ValueError("Expected a two- or three-dimensional image.")
    if image.dtype != np.uint8:
        raise ValueError("Expected an image with dtype uint8.")
    if image.size == 0:
        raise ValueError("Expected a non-empty image.")

    height, width = image.shape[:2]
    if right > width:
        raise ValueError("Right exceeds image width.")
    if bottom > height:
        raise ValueError("Bottom exceeds image height.")
    if left >= right:
        raise ValueError("Left must be less than right.")
    if top >= bottom:
        raise ValueError("Top must be less than bottom.")
    if left < 0:
        raise ValueError("Left must be non-negative.")
    if top < 0:
        raise ValueError("Top must be non-negative.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Expected exactly three color channels for RGB.")

    cropped_array = image[top:bottom, left:right].copy()
    return cropped_array

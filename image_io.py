from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

MODE_RGBA = "RGBA"
SUPPORTED_FORMATS = ("JPEG", "PNG")


def load_image(path: Path) -> NDArray[np.uint8]:
    """Load an image as an independent RGB array."""
    # Open the image using a context manager.
    with Image.open(path) as image:
        if image.format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image.format}")
        oriented_image = ImageOps.exif_transpose(image)

        rgba_image = oriented_image.convert(MODE_RGBA)
        background = Image.new(
            MODE_RGBA,
            rgba_image.size,
            (255, 255, 255, 255),
        )
        composited_image = Image.alpha_composite(background, rgba_image)
        rgb_image = composited_image.convert("RGB")

    # Create an independent NumPy array.
    array = np.array(rgb_image, dtype=np.uint8, copy=True)
    # Return the array.
    return array


def save_image(image: NDArray[np.uint8], path: Path) -> None:
    """Save an image array as a PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image)
    pil_image.save(path, format="PNG")

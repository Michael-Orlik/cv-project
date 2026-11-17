import numpy as np
import pytest

from preprocessing import binarize, binarize_adaptive


def test_binarize_separates_dark_and_light_pixels():
    """Produce a binary mask while preserving the source grayscale data."""
    image = np.array([[0, 10], [200, 255]], dtype=np.uint8)
    original = image.copy()

    result = binarize(image)

    expected = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(image, original)


def test_binarize_adaptive_handles_bright_and_shadowed_regions():
    """Preserve dark marks against both bright and shadowed backgrounds."""
    image = np.full((101, 201), 220, dtype=np.uint8)
    image[:, 100:] = 120
    image[50, 50] = 20
    image[50, 150] = 20
    original = image.copy()

    result = binarize_adaptive(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})
    assert result[20, 20] == 255
    assert result[20, 180] == 255
    assert result[50, 50] == 0
    assert result[50, 150] == 0
    np.testing.assert_array_equal(image, original)


def test_binarize_adaptive_rejects_even_window():
    """Reject an even window size with a clear validation error."""
    image = np.full((3, 3), 200, dtype=np.uint8)

    with pytest.raises(ValueError, match="Block size must be odd"):
        binarize_adaptive(image, block_size=50)

import numpy as np

from preprocessing import binarize


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

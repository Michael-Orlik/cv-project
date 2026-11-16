import numpy as np
import pytest

from image_ops import crop_image, to_grayscale


def test_to_grayscale_converts_primary_colors():
    """Convert red, green and blue pixels to expected grayscale values."""
    image = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
        dtype=np.uint8,
    )
    expected = np.array([[76, 150, 29]], dtype=np.uint8)

    result = to_grayscale(image)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


def test_crop_image_selects_requested_region():
    """Select the expected rows and columns."""
    image = np.array(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [11, 12],
            [21, 22],
        ],
        dtype=np.uint8,
    )

    result = crop_image(
        image,
        left=1,
        top=0,
        right=3,
        bottom=2,
    )

    np.testing.assert_array_equal(result, expected)


def test_crop_image_returns_independent_data():
    """Keep the source image unchanged when the crop is modified."""
    image = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.uint8,
    )

    result = crop_image(
        image,
        left=0,
        top=0,
        right=1,
        bottom=1,
    )

    result[0, 0] = 255

    assert image[0, 0] == 10


def test_to_grayscale_returns_independent_data():
    """Keep the RGB source unchanged when grayscale data is modified."""
    image = np.array([[[255, 0, 0]]], dtype=np.uint8)
    result = to_grayscale(image)
    result[0, 0] = 0
    np.testing.assert_array_equal(image[0, 0], [255, 0, 0])


def test_crop_image_rejects_negative_left():
    """Reject a crop that starts before the image's left edge."""
    image = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Left must be non-negative"):
        crop_image(
            image,
            left=-1,
            top=0,
            right=1,
            bottom=1,
        )

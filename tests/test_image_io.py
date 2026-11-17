import numpy as np
import pytest
from PIL import ExifTags, Image, UnidentifiedImageError

from image_io import load_image, save_image


def test_load_image_returns_rgb_array(tmp_path):
    """Load an RGB image with the expected dimensions and pixel values."""
    path = tmp_path / "sample.png"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(path)

    result = load_image(path)

    assert result.shape == (2, 3, 3)
    assert result.dtype == np.uint8
    assert result.flags.owndata
    assert result.flags.writeable
    np.testing.assert_array_equal(result[0, 0], [10, 20, 30])


@pytest.mark.parametrize(
    ("rgba", "expected"),
    [
        ((0, 0, 0, 0), [255, 255, 255]),
        ((255, 0, 0, 128), [255, 127, 127]),
        ((10, 20, 30, 255), [10, 20, 30]),
    ],
)
def test_load_image_composites_alpha_over_white(tmp_path, rgba, expected):
    """Composite transparent pixels over a white background."""
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (1, 1), rgba).save(path)

    result = load_image(path)

    np.testing.assert_array_equal(result[0, 0], expected)


def test_load_image_applies_exif_orientation(tmp_path):
    """Rotate pixels according to the EXIF orientation."""
    pixels = np.array(
        [[[255, 0, 0], [0, 0, 255]]],
        dtype=np.uint8,
    )
    path = tmp_path / "oriented.png"

    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    Image.fromarray(pixels).save(path, exif=exif)

    result = load_image(path)

    assert result.shape == (2, 1, 3)
    np.testing.assert_array_equal(result[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(result[1, 0], [0, 0, 255])


def test_load_image_reads_jpeg(tmp_path):
    """Load JPEG pixels within the expected compression tolerance."""
    path = tmp_path / "sample.jpg"
    color = [32, 128, 224]
    Image.new("RGB", (4, 3), tuple(color)).save(
        path, quality=95, subsampling=0
    )

    result = load_image(path)

    assert result.shape == (3, 4, 3)
    assert result.dtype == np.uint8
    expected = np.broadcast_to(color, result.shape)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2)


def test_load_image_converts_grayscale_png_to_rgb(tmp_path):
    """Preserve grayscale brightness in all three RGB channels."""
    pixels = np.array([[0, 64, 255], [32, 128, 200]], dtype=np.uint8)
    path = tmp_path / "grayscale.png"
    Image.fromarray(pixels).save(path)

    result = load_image(path)

    expected = np.stack([pixels, pixels, pixels], axis=-1)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


def test_load_image_handles_palette_transparency(tmp_path):
    """Expand palette colors and composite the transparent entry on white."""
    path = tmp_path / "palette.png"
    palette_image = Image.new("P", (3, 1))
    palette_image.putpalette(
        [0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * 759
    )
    palette_image.putdata([0, 1, 2])
    palette_image.save(path, transparency=0)

    result = load_image(path)

    expected = np.array(
        [[[255, 255, 255], [255, 0, 0], [0, 255, 0]]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


def test_load_image_rejects_missing_file(tmp_path):
    """Report a missing input file."""
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_load_image_rejects_damaged_file(tmp_path):
    """Reject a file whose content is not an image."""
    path = tmp_path / "damaged.png"
    path.write_bytes(b"not an image")

    with pytest.raises(UnidentifiedImageError):
        load_image(path)


def test_load_image_rejects_unsupported_format(tmp_path):
    """Reject unsupported content even when the filename ends in PNG."""
    path = tmp_path / "disguised.png"
    Image.new("RGB", (3, 2)).save(path, format="GIF")

    with pytest.raises(ValueError, match="^Unsupported image format: GIF$"):
        load_image(path)


def test_save_image_writes_binary_png_in_nested_directory(tmp_path):
    """Create parent directories and preserve the binary pixel values."""
    image = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    path = tmp_path / "nested" / "binary.png"

    save_image(image, path)

    with Image.open(path) as saved_image:
        assert saved_image.format == "PNG"
        np.testing.assert_array_equal(np.array(saved_image), image)

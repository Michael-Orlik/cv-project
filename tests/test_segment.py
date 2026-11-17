import numpy as np

from segment import find_horizontal_bounds, find_line_boxes


def test_horizontal_bounds_keep_punctuation_and_ignore_faint_marks():
    mask = np.zeros((40, 200), dtype=np.uint8)
    grayscale = np.full(mask.shape, 255, dtype=np.uint8)
    regions = ((20, 35, 8, 30), (45, 65, 8, 30), (78, 81, 25, 28))
    for left, right, top, bottom in regions:
        mask[top:bottom, left:right] = 255
        grayscale[top:bottom, left:right] = 0

    # Faint writing and a single dark speck should not extend the text bounds.
    mask[10:30, 170:175] = 255
    grayscale[10:30, 170:175] = 220
    mask[15, 1] = 255
    grayscale[15, 1] = 0

    assert find_horizontal_bounds(mask, grayscale, padding=5) == (15, 86)


def test_horizontal_bounds_ignore_an_isolated_edge_artifact():
    mask = np.zeros((40, 300), dtype=np.uint8)
    mask[8:30, 80:100] = 255
    mask[8:30, 115:135] = 255
    mask[10:25, 295:300] = 255
    grayscale = 255 - mask

    assert find_horizontal_bounds(mask, grayscale, padding=5) == (75, 140)


def test_horizontal_bounds_preserve_words_separated_by_a_large_space():
    mask = np.zeros((40, 500), dtype=np.uint8)
    mask[8:30, 20:60] = 255
    mask[8:30, 350:370] = 255

    assert find_horizontal_bounds(mask, 255 - mask, padding=5) == (15, 375)


def test_horizontal_bounds_keep_full_width_when_no_ink_is_found():
    mask = np.zeros((30, 100), dtype=np.uint8)
    grayscale = np.full(mask.shape, 255, dtype=np.uint8)

    assert find_horizontal_bounds(mask, grayscale) == (0, 100)


def test_line_boxes_trim_two_rows_without_modifying_source_pixels():
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    for top in (40, 150):
        for left in (70, 105, 140):
            image[top:top + 30, left:left + 20] = 0
    original = image.copy()

    boxes = find_line_boxes(image)

    assert len(boxes) == 2
    for (left, top, right, bottom), ink_top in zip(boxes, (40, 150)):
        assert 0 < left <= 70
        assert 160 <= right < image.shape[1]
        assert top <= ink_top < ink_top + 30 <= bottom
    assert boxes[0][3] <= boxes[1][1]
    np.testing.assert_array_equal(image, original)

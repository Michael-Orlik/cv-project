# Handwritten Text Recognition to Markdown

A Python application for converting images of handwritten text into Markdown.

The current implementation loads JPEG and PNG images, applies EXIF
orientation and composites transparency over a white background.
It converts images to RGB and grayscale arrays and displays their metadata.
It can also export a binary PNG using Otsu or adaptive Gaussian thresholding.

The application can recognize a manually cropped English handwritten line.
Markdown export is planned for a later stage.

## Requirements

- Python 3.12
- uv

## Setup

Run from the project root:

```bash
uv sync --locked
```

This prepares the `.venv` environment and installs the dependencies
recorded in `uv.lock`, including development tools.

## Usage

Display the command-line help:

```bash
uv run python main.py --help
```

Provide an existing image and a path for the future Markdown output:

```bash
uv run python main.py note.jpg result.md
```

Example output for an image with width 1200 and height 800:

```text
Grayscale shape: (800, 1200)
RGB shape: (800, 1200, 3)
Input image: note.jpg
Output Markdown: result.md
Image size: 1200 x 800
RGB dtype: uint8
```

## Binary image

Save a binary version of an image with black foreground and white background.

```bash
uv run python main.py note.jpg result.md --binary-output output/binary.png
```

The parent directory is created as needed. The program reports the saved path.

Otsu uses one global threshold for the entire image. Uneven lighting can turn
shadows into black regions, and notebook grid lines can appear in the mask.
Inspect the result before using it for text-line segmentation.

## Adaptive thresholding

Use local Gaussian thresholds to process an image with uneven lighting.

```bash
uv run python main.py note.jpg result.md --method adaptive \
    --block-size 51 --c 10 --binary-output output/adaptive.png
```

- `--method` selects `otsu` or `adaptive`. The default is `otsu`.
- `--block-size` sets the local window size. It must be an odd integer of
  at least 3. The default is 51.
- `--c` is the finite constant subtracted from the local weighted mean.
  The default is 10.

Compare the saved image with the Otsu result. Adaptive thresholding can reduce
black shadow regions. Notebook grid lines and small background artifacts may
still appear in the output.

## OCR preparation

Crop one RGB text line and load the TrOCR processor and model:

```bash
uv run python main.py note.jpg result.md --prepare-ocr \
    --crop 145 50 1800 185 --line-output output/line.png
```

Choose crop coordinates for your image after EXIF orientation.
The order is left, top, right, bottom; right and bottom are exclusive.
Inspect the saved PNG to confirm that the complete line is visible.

The first run downloads checkpoint files to the Hugging Face cache.
Later runs reuse cached files.

- Model: `microsoft/trocr-base-handwritten`
- Revision: `eaacaf452b06415df8f10bb6fad3a4c11e609406`
- Device: CPU
- Model dtype: float32
- Evaluation mode enabled

This checkpoint is intended for English handwritten text lines.
Successful preparation ends with `OCR resources ready.`

## Line recognition

Add `--recognize-line` to recognize the prepared RGB crop:

```bash
uv run python main.py note.jpg result.md --prepare-ocr --recognize-line \
    --crop 145 50 1800 185 --line-output output/line.png
```

The recognized text is printed after `Recognized text:`.
Recognition may contain errors, including missing punctuation.
Markdown export is planned for a later stage.

## Page segmentation

Detect candidate text lines and save their RGB crops:

```bash
uv run python main.py note.jpg result.md \
    --segment-page --lines-output output/lines
```

The program saves files named `line-01.png`, `line-02.png`, and so on,
ordered from top to bottom.

This experimental method uses horizontal pixel density and assumes
a single-column page. Each crop spans the full image width.
Inspect the crops before recognition: boundaries may cut letter strokes
or include parts of neighboring lines.

This mode saves images only. Page recognition and Markdown export
are planned for a later stage.

## Tests

Run all tests from the project root:

```bash
uv run python -m pytest
```

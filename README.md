# Handwritten Text Recognition to Markdown

A Python application for converting images of handwritten text into Markdown.

The current implementation loads JPEG and PNG images, applies EXIF
orientation and composites transparency over a white background.
It converts images to RGB and grayscale arrays and displays their metadata.

Handwriting recognition and Markdown export are planned for later stages.

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

## Tests

Run all tests from the project root:

```bash
uv run python -m pytest
```

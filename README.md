# Handwritten Text Recognition to Markdown

A Python application for converting images of handwritten text into Markdown.

The current implementation accepts input and output paths through a
command-line interface and displays them in the terminal.
Image processing and text recognition are planned for later stages.

## Requirements

- Python 3.12
- uv

## Setup

Run the following command from the project root:

```bash
uv sync --locked
```

This creates the `.venv` environment as needed and installs the dependencies
recorded in `uv.lock`.

## Usage

Display the command-line help:

```bash
uv run python main.py --help
```

Pass an input image path and an output Markdown path:

```bash
uv run python main.py note.jpg result.md
```

Expected output:

```text
Input image: note.jpg
Output Markdown: result.md
```
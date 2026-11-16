import argparse
import sys
from pathlib import Path

from image_io import load_image
from image_ops import to_grayscale


def main() -> int:
    """Load an image and display RGB and grayscale metadata."""
    parser = argparse.ArgumentParser(
        description="Inspect image data for handwritten text recognition."
    )

    # Positional arguments are assigned in the order they are declared.
    parser.add_argument(
        "input_image",
        type=Path,
        help="Path to the input image.",
    )

    parser.add_argument(
        "output_md",
        help="Path to the output Markdown file.",
    )

    # Parse command-line arguments into named values.
    args = parser.parse_args()
    # Load the image as an RGB array.
    try:
        image = load_image(args.input_image)
    except FileNotFoundError:
        print(
            f"Error: input image not found: {args.input_image}",
            file=sys.stderr,
        )
        return 1

    except OSError as error:
        print(
            f"Error: cannot read input image: {args.input_image} ({error})",
            file=sys.stderr,
        )
        return 1

    except ValueError as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1

    # Create a separate grayscale array.
    grayscale_image = to_grayscale(image)
    # NumPy stores image dimensions in height, width order.
    height, width = image.shape[:2]
    print(f"Grayscale shape: {grayscale_image.shape}")
    print(f"RGB shape: {image.shape}")
    print(f"Input image: {args.input_image}")
    print(f"Output Markdown: {args.output_md}")
    print(f"Image size: {width} x {height}")
    print(f"RGB dtype: {image.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import sys
from pathlib import Path

from image_io import load_image, save_image
from image_ops import to_grayscale
from preprocessing import binarize, binarize_adaptive


def main() -> int:
    """Inspect image data and optionally save a binary PNG."""
    parser = argparse.ArgumentParser(
        description="Inspect image data and optionally save a binary PNG."
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

    parser.add_argument(
        "--binary-output",
        type=Path,
        help="Path to save the binary PNG image.",
    )

    parser.add_argument(
        "--method",
        choices=("otsu", "adaptive"),
        default="otsu",
        help="Binarization method.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=51,
        help="Odd window size for adaptive thresholding.",
    )

    parser.add_argument(
        "--c",
        type=float,
        default=10.0,
        help="Constant subtracted from the local threshold.",
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
    
    # Select the requested binarization method.
    # Create a binary adaptive image
    try:
        if args.method == "adaptive":
            binary_image = binarize_adaptive(
                grayscale_image,
                block_size=args.block_size,
                c=args.c,
            )
        else:
            binary_image = binarize(grayscale_image)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # NumPy stores image dimensions in height, width order.
    height, width = image.shape[:2]
    print(f"Grayscale shape: {grayscale_image.shape}")
    print(f"RGB shape: {image.shape}")
    print(f"Input image: {args.input_image}")
    print(f"Output Markdown: {args.output_md}")
    print(f"Image size: {width} x {height}")
    print(f"RGB dtype: {image.dtype}")

    if args.binary_output:
        try:
            save_image(binary_image, args.binary_output)
        except OSError as error:
            print(f"Error: cannot save binary image: {error}", file=sys.stderr)
            return 1

        print(f"Binary image: {args.binary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

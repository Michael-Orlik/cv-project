import argparse
import sys
from pathlib import Path

from image_io import load_image, save_image
from image_ops import crop_image, to_grayscale
from preprocessing import binarize, binarize_adaptive


def main() -> int:
    """Run image preprocessing or prepare a text line for OCR.

    Returns
    -------
    int
        0 on success or 1 when an image operation or model loading fails.

    Raises
    ------
    SystemExit
        If argparse displays help (0) or rejects command-line arguments (2).

    Notes
    -----
    OCR preparation saves an RGB crop and loads the pinned model.
    Text recognition and Markdown export are not implemented yet.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess an image or prepare a text line for OCR."
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

    # OCR preparation.
    parser.add_argument(
        "--prepare-ocr",
        action="store_true",
        help="Prepare a line image and load OCR resources.",
    )

    # Require four coordinates in the EXIF-oriented image coordinate system.
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Bounds of one text line; right and bottom are exclusive.",
    )

    parser.add_argument(
        "--line-output",
        type=Path,
        help="Path to save the cropped RGB line as PNG.",
    )

    # Parse command-line arguments into named values.
    args = parser.parse_args()

    # Reject incomplete preparation settings before accessing files.
    if args.prepare_ocr:
        if args.crop is None or args.line_output is None:
            parser.error("--prepare-ocr requires --crop and --line-output.")
    elif args.crop is not None or args.line_output is not None:
        parser.error("--crop and --line-output require --prepare-ocr.")

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

    # RGB fragment preparation.
    if args.prepare_ocr:
        # Unpack coordinates in the same order as the CLI arguments.
        left, top, right, bottom = args.crop

        try:
            # Preserve the source image when choosing the crop destination
            if args.line_output.exists() and args.line_output.samefile(
                args.input_image
            ):
                raise ValueError("Line output must differ from input image.")

            # Existing crop validation checks image bounds and coordinates.
            line_image = crop_image(
                image,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
            save_image(line_image, args.line_output)
        except ValueError as error:
            print(f"Error: cannot prepare line image: {error}", file=sys.stderr)
            return 1
        except OSError as error:
            print(f"Error: cannot save line image: {error}", file=sys.stderr)
            return 1

        print(f"Line image: {args.line_output}")
        print(f"Line shape: {line_image.shape}")

        # Load OCR dependencies after validating and saving the line image.
        try:
            from recognize import MODEL_ID, MODEL_REVISION, load_recognizer

            processor, model = load_recognizer()
        except (OSError, ValueError, RuntimeError) as error:
            # Report expected loading failures through the CLI error channel.
            print(
                f"Error: cannot load OCR resources: {error}",
                file=sys.stderr,
            )
            return 1

        print(f"Model: {MODEL_ID}")
        print(f"Revision: {MODEL_REVISION}")
        print(f"Device: {model.device}")
        print(f"Processor: {type(processor).__name__}")
        print(f"Evaluation mode: {not model.training}")
        print("OCR resources ready.")

        return 0

    # Create a separate grayscale array.
    grayscale_image = to_grayscale(image)

    # Select the requested binarization method
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

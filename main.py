import argparse
import sys
from pathlib import Path

import cv2

from image_io import load_image, save_image
from image_ops import crop_image, to_grayscale
from preprocessing import binarize, binarize_adaptive
from segment import find_line_boxes


def save_markdown(lines: list[str], path: Path) -> None:
    """Save recognized lines as UTF-8 Markdown text.

    Raises:
        OSError: If the destination directory or file cannot be written.
    """

    # Join recognized lines in their reading order.
    content = "\n".join(lines)
    if content:
        content += "\n"

    # Create the destination directory if it does not exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the text using an explicit encoding.
    path.write_text(content, encoding="utf-8")


def main() -> int:
    """Run image processing or OCR with Markdown output.

    Return 0 on success or 1 if processing, recognition, or saving fails.

    Raises:
        SystemExit: If argparse displays help (0) or rejects arguments (2).
    """
    parser = argparse.ArgumentParser(
        description="Preprocess images, segment pages, or save OCR text as Markdown."
    )

    # Positional arguments are assigned in the order they are declared.
    parser.add_argument(
        "input_image",
        type=Path,
        help="Path to the input image.",
    )

    parser.add_argument(
        "output_md",
        type=Path,
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

    parser.add_argument(
        "--recognize-line",
        action="store_true",
        help="Recognize the cropped line after OCR preparation.",
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

    parser.add_argument(
        "--segment-page",
        action="store_true",
        help="Detect text-line regions and save their RGB crops.",
    )

    parser.add_argument(
        "--recognize-page",
        action="store_true",
        help="Recognize all text lines detected during page segmentation.",
    )

    parser.add_argument(
        "--lines-output",
        type=Path,
        help="Directory for automatically cropped line images.",
    )

    # Parse command-line arguments into named values.
    args = parser.parse_args()

    # Page recognition uses the line regions found during segmentation.
    if args.recognize_page and not args.segment_page:
        parser.error("--recognize-page requires --segment-page.")

    # Manual cropping and automatic segmentation are separate modes.
    if args.segment_page and args.prepare_ocr:
        parser.error("--segment-page and --prepare-ocr cannot be used together.")

    # Automatic segmentation requires a destination directory
    if args.segment_page:
        if args.lines_output is None:
            parser.error("--segment-page requires --lines-output.")
    elif args.lines_output is not None:
        parser.error("--lines-output requires --segment-page.")

    # Recognition requires the crop and model created during preparation.
    if args.recognize_line and not args.prepare_ocr:
        parser.error("--recognize-line requires --prepare-ocr.")

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

    if args.recognize_page or args.recognize_line:
        try:
            # Protect the source image from being overwritten by text.
            if args.output_md.exists() and args.output_md.samefile(
                args.input_image
            ):
                raise ValueError(
                    "Markdown output must differ from the input image."
                )
        except (OSError, ValueError) as error:
            print(
                f"Error: invalid Markdown destination: {error}",
                file=sys.stderr,
            )
            return 1

    if args.segment_page:
        try:
            boxes = find_line_boxes(image)

            for index, (left, top, right, bottom) in enumerate(boxes, start=1):
                # Crop RGB pixels from the source image using detected bounds.
                line_image = crop_image(
                    image,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )

                line_path = args.lines_output / f"line-{index:02d}.png"

                # Prevent a generated crop from overwriting the source image.
                if line_path.exists() and line_path.samefile(args.input_image):
                    raise ValueError("Line output must differ from input image.")

                save_image(line_image, line_path)

        except (ValueError, OSError, cv2.error) as error:
            print(
                f"Error: cannot segment page or save line images: {error}",
                file=sys.stderr,
            )
            return 1

        print(f"Saved lines: {len(boxes)}")
        print(f"Lines output: {args.lines_output}")

        if args.recognize_page:
            if not boxes:
                try:
                    save_markdown([], args.output_md)
                except OSError as error:
                    print(
                        f"Error: cannot save Markdown: {error}",
                        file=sys.stderr,
                    )
                    return 1

                print("No text lines found.")
                print(f"Markdown saved: {args.output_md}")
                return 0

            try:
                from recognize import load_recognizer, recognize_line

                # Load one processor and model for all lines on this page.
                processor, model = load_recognizer()
            except (OSError, ValueError, RuntimeError) as error:
                print(
                    f"Error: cannot load OCR resources: {error}",
                    file=sys.stderr,
                )
                return 1

            print("Page OCR resources ready.")

            recognized_lines = []

            for index, (left, top, right, bottom) in enumerate(boxes, start=1):
                try:
                    # Read RGB pixels from the current page using detected bounds.
                    line_image = crop_image(
                        image,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                    )

                    # Reuse the same processor and model for every line.
                    text = recognize_line(line_image, processor, model)

                except (ValueError, RuntimeError) as error:
                    print(
                        f"Error: cannot recognize line {index}: {error}",
                        file=sys.stderr,
                    )
                    return 1

                recognized_lines.append(text)
                print(f"Line {index}: {text}")

            print(f"Recognized lines: {len(recognized_lines)}")

            # Save the page text after all detected lines have been recognized.
            try:
                save_markdown(recognized_lines, args.output_md)
            except OSError as error:
                print(
                    f"Error: cannot save Markdown: {error}",
                    file=sys.stderr,
                )
                return 1

            print(f"Markdown saved: {args.output_md}")

        return 0

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
            from recognize import (
                MODEL_ID,
                MODEL_REVISION,
                load_recognizer,
                recognize_line,
            )

            processor, model = load_recognizer()
        except (OSError, ValueError, RuntimeError) as error:
            # Report expected loading failures through the CLI error channel
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

        if args.recognize_line:
            try:
                # Reuse the RGB crop and the loaded model.
                text = recognize_line(line_image, processor, model)
            except (ValueError, RuntimeError) as error:
                print(
                    f"Error: cannot recognize line: {error}",
                    file=sys.stderr,
                )
                return 1

            print(f"Recognized text: {text}")

            try:
                save_markdown([text], args.output_md)
            except OSError as error:
                print(
                    f"Error: cannot save Markdown: {error}",
                    file=sys.stderr,
                )
                return 1

            print(f"Markdown saved: {args.output_md}")
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
            if args.binary_output.exists() and args.binary_output.samefile(
                args.input_image
            ):
                raise ValueError("Binary output must differ from input image.")
            save_image(binary_image, args.binary_output)
        except (OSError, ValueError) as error:
            print(f"Error: cannot save binary image: {error}", file=sys.stderr)
            return 1

        print(f"Binary image: {args.binary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

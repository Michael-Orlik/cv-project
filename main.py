import argparse


def main():
    """Parse command-line arguments and display input and output paths."""
    parser = argparse.ArgumentParser(
        description="Recognize handwritten text and convert it to Markdown."
    )

    # Positional arguments are assigned in the order they are declared.
    parser.add_argument(
        "input_image",
        help="Path to the input image.",
    )
    parser.add_argument(
        "output_md",
        help="Path to the output Markdown file.",
    )

    args = parser.parse_args()

    print(f"Input image: {args.input_image}")
    print(f"Output Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
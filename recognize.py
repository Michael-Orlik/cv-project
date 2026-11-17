"""Configure TrOCR resources for line-image OCR preparation.

The processor and model must use the same checkpoint revision so that
their configuration, tokenizer and weights remain aligned.
"""

import torch

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Checkpoint selected for the initial handwritten-line experiment.
MODEL_ID = "microsoft/trocr-base-handwritten"

# Pin an exact model repository commit for reproducible loading.
MODEL_REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"

# Use CPU as the baseline device for the first verified implementation.
DEVICE = "cpu"


def load_recognizer() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the pinned TrOCR processor and model for CPU inference.

    Returns
    -------
    processor: TrOCRProcessor
        Image processor and tokenizer from the pinned checkpoint.
    model: VisionEncoderDecoderModel
        Float32 model placed on DEVICE and configured for evaluation.

    Raises
    ------
    OSError
        If checkpoint files cannot be retrieved or read.
    ValueError
        If the checkpoint configuration is incompatible with the loader.
    RuntimeError
        If PyTorch fails to initialize or place the model parameters.

    Notes
    -----
    The first call may download checkpoint files to the Hugging Face cache.
    Both resources use MODEL_REVISION to keep their artifacts aligned.
    Call this function once per preparation run and reuse its results.
    """
    # Load preprocessing and tokenization settings from the pinned revision.
    processor = TrOCRProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    # Load matching weights with an explicit dtype and serialization format.
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        use_safetensors=True,
    )

    # Place the model on the baseline device used by this implementation.
    model.to(DEVICE)

    # Configure layers such as dropout for evaluation.
    model.eval()

    return processor, model
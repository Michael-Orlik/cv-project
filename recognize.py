"""Configure TrOCR resources for line-image OCR preparation.

The processor and model must use the same checkpoint revision so that
their configuration, tokenizer and weights remain aligned.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# Checkpoint selected for the initial handwritten-line experiment.
MODEL_ID = "microsoft/trocr-large-handwritten"

# Pin an exact model repository commit for reproducible loading.
MODEL_REVISION = "e68501f437cd2587ae5d68ee457964cac824ddee"

# Use CPU as the baseline device for the first verified implementation.
DEVICE = "cpu"


def load_recognizer() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load the pinned processor and float32 model on CPU in evaluation mode.

    Checkpoint files are reused from the Hugging Face cache.

    Raises:
        OSError: If checkpoint files cannot be downloaded or read.
        ValueError: If the checkpoint configuration is incompatible.
        RuntimeError: If PyTorch cannot initialize the model.
    """
    # Load preprocessing and tokenization settings from the pinned revision.
    processor = TrOCRProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    # This checkpoint provides PyTorch weights in pytorch_model.bin.
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        use_safetensors=False,
    )

    # Place the model on the baseline device used by this implementation.
    model.to(DEVICE)

    # Configure layers such as dropout for evaluation.
    model.eval()

    return processor, model


def recognize_line(
    image: NDArray[np.uint8],
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
) -> str:
    """Recognize text in a cropped RGB line image."""
    # Prepare the RGB image and move its tensor to the model device.
    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(model.device)

    # Generate token IDs without tracking gradients.
    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )

    # Convert the first prediction to text and remove special tokens.
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return text.strip()

"""Local baidu/Unlimited-OCR adapter using the model's custom Transformers API.

Requires CUDA and BF16 support. See README_UNLIMITED.md for setup commands.
Reference: https://huggingface.co/baidu/Unlimited-OCR
"""

from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile


DEFAULT_MODEL = "baidu/Unlimited-OCR"


def extract_text(
    image_path: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    mode: str = "gundam",
    max_length: int = 32768,
) -> dict:
    """Read one image locally; return the model's text, including layout tags.

    Heavy dependencies load only when this engine is selected. Model weights
    and custom Python code are downloaded from Hugging Face on first use.
    """
    from PIL import Image

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if mode not in ("gundam", "base"):
        raise ValueError("Unlimited OCR mode must be gundam or base.")
    if not 1 <= max_length <= 32768:
        raise ValueError("Unlimited OCR max length must be between 1 and 32768.")
    if path.suffix.lower() == ".pdf":
        raise ValueError("This Unlimited OCR adapter accepts images; convert PDF pages to images first.")
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) > 1:
            raise ValueError("Supply a single-frame image for Unlimited OCR.")
        original_size = list(image.size)
        image.verify()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Unlimited-OCR requires an NVIDIA CUDA GPU and CUDA-enabled PyTorch. "
            "This Python environment has no available CUDA GPU. See README_UNLIMITED.md."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Unlimited-OCR requires a CUDA GPU with BF16 support.")

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model} on CUDA (first use downloads the model)...", file=sys.stderr, flush=True)
    # The upstream infer method creates scratch folders even with saving off.
    # Keep them temporary; main.py owns the final text/JSON output paths.
    with redirect_stdout(sys.stderr):
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        ocr = AutoModel.from_pretrained(
            model, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        try:
            with tempfile.TemporaryDirectory(prefix="unlimited_ocr_") as scratch:
                with torch.inference_mode():
                    text = ocr.infer(
                        tokenizer,
                        prompt="<image>document parsing.",
                        image_file=str(path), output_path=scratch,
                        base_size=1024, image_size=640 if mode == "gundam" else 1024,
                        crop_mode=mode == "gundam", max_length=max_length,
                        no_repeat_ngram_size=35, ngram_window=128,
                        save_results=False, eval_mode=True,
                    )
        finally:
            del ocr
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Unlimited-OCR returned no text. Try a clearer or cropped image.")
    return {
        "engine": "unlimited", "model": model, "image": str(path),
        "text": text.strip(), "original_size": original_size,
        "mode": mode, "max_length": max_length,
    }

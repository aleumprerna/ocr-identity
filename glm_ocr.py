"""Local GLM-OCR text extraction through Ollama's vision API.

Setup: install/start Ollama, then run `ollama pull glm-ocr:q8_0` once.
Usage from Python: `from glm_ocr import extract_text`.
Model: https://github.com/zai-org/GLM-OCR
API: https://docs.ollama.com/api/chat
"""

import base64
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_MODEL = "glm-ocr:q8_0"
DEFAULT_HOST = "http://127.0.0.1:11434"


def extract_text(
    image_path: Path | str,
    *,
    model: str | None = None,
    host: str | None = None,
    timeout: float = 600,
) -> dict:
    """Read an image with GLM-OCR; return text and generation metadata.

    PaddleOCR and Tesseract are not loaded. Ollama handles model inference.
    Input is converted to PNG, with the longest side capped at 1600 pixels.
    The default endpoint is local; a supplied host receives the image instead.
    """
    from PIL import Image, ImageOps

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if timeout <= 0:
        raise ValueError("GLM timeout must be greater than zero.")
    model = model or os.environ.get("GLM_OCR_MODEL", "").strip() or DEFAULT_MODEL
    host = (host or os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST).rstrip("/")
    if "://" not in host:
        host = "http://" + host
    endpoint = urlsplit(host)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError("GLM host must be an HTTP(S) Ollama server URL.")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("GLM host must not include credentials, a query, or a fragment.")

    with Image.open(path) as original:
        if getattr(original, "n_frames", 1) > 1:
            raise ValueError("GLM image extraction accepts one frame. Supply a single-page image.")
        image = ImageOps.exif_transpose(original).convert("RGB")
        original_size = list(image.size)
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        input_size = list(image.size)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Text Recognition:",
            "images": [base64.b64encode(buffer.getvalue()).decode("ascii")],
        }],
        "stream": False,
        # Bound the context allocation rather than using the model's much
        # larger maximum. Unload after completion to release local memory.
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 2048},
        "keep_alive": 0,
    }
    request = Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f"GLM model not found on Ollama. Run: ollama pull {model}. "
                "Also check --glm-host points to an Ollama server."
            ) from None
        raise RuntimeError(
            f"Ollama HTTP {exc.code}: could not run GLM-OCR. "
            "Check Ollama's server logs and available memory."
        ) from None
    except TimeoutError:
        raise RuntimeError(
            "GLM-OCR timed out. CPU inference can be slow; retry with a cropped "
            "image or increase --glm-timeout."
        ) from None
    except URLError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {host}. Start the Ollama app or run "
            f"'ollama serve', then install the model with 'ollama pull {model}'."
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Ollama returned an invalid JSON response.") from None

    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned an invalid GLM response.")
    if result.get("error"):
        raise RuntimeError("Ollama reported a GLM inference error. Check the server logs.")
    message = result.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("GLM-OCR returned no text. Try a clearer or cropped image.")
    if result.get("done") is not True:
        raise RuntimeError("Ollama returned an incomplete GLM response.")
    return {
        "engine": "glm",
        "model": result.get("model", model),
        "image": str(path),
        "text": text.strip(),
        "original_size": original_size,
        "input_size": input_size,
        "truncated": result.get("done_reason") == "length",
        "usage": {
            "prompt_tokens": result.get("prompt_eval_count"),
            "generated_tokens": result.get("eval_count"),
            "total_duration_ns": result.get("total_duration"),
        },
    }

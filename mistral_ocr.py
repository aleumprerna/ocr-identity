"""Mistral hosted OCR adapter, separate from local model implementations.

Set MISTRAL_API_KEY. Selected files are sent to https://api.mistral.ai/v1/ocr.
Reference: https://docs.mistral.ai/api/endpoint/ocr
"""

import base64
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def extract_text(file_path: Path | str, *, model: str | None = None, timeout: float = 120) -> dict:
    """Extract page Markdown from a local image or PDF with Mistral OCR."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        raise RuntimeError('MISTRAL_API_KEY is not set. In PowerShell: $env:MISTRAL_API_KEY = "your_api_key"')
    if timeout <= 0:
        raise ValueError("Mistral timeout must be greater than zero.")
    mime_types = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/png", ".pdf": "application/pdf",
    }
    suffix = path.suffix.lower()
    if suffix not in mime_types:
        raise ValueError("Mistral input must be PNG, JPEG, WebP, or PDF.")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("This integration limits Base64 inputs to 50 MiB; split the file first.")
    content = path.read_bytes()
    if not content:
        raise ValueError("The input file is empty.")
    if suffix == ".webp":
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(content)) as image:
            if getattr(image, "n_frames", 1) > 1:
                raise ValueError("Animated WebP is not supported; supply one image or a PDF.")
            buffer = io.BytesIO()
            ImageOps.exif_transpose(image).convert("RGB").save(buffer, format="PNG")
            content = buffer.getvalue()
    encoded = base64.b64encode(content).decode("ascii")
    document_type = "document_url" if suffix == ".pdf" else "image_url"
    selected_model = model or os.environ.get("MISTRAL_OCR_MODEL", "").strip() or "mistral-ocr-latest"
    payload = {
        "model": selected_model,
        "document": {"type": document_type, document_type: f"data:{mime_types[suffix]};base64,{encoded}"},
        "include_image_base64": False,
    }
    request = Request(
        "https://api.mistral.ai/v1/ocr", data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        advice = {
            401: "Check MISTRAL_API_KEY.", 403: "Check this key's OCR model access.",
            402: "Check account credits or billing.", 413: "Split the input into smaller files.",
            422: "Check the file format and model name.", 429: "Check account quota and retry later.",
        }.get(exc.code, "Check the Mistral service and retry later.")
        raise RuntimeError(f"Mistral OCR HTTP {exc.code}. {advice}") from None
    except TimeoutError:
        raise RuntimeError("Mistral OCR timed out. Try a smaller document or retry later.") from None
    except URLError:
        raise RuntimeError("Cannot connect to Mistral OCR. Check internet and proxy settings.") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Mistral OCR returned an invalid JSON response.") from None
    pages = result.get("pages") if isinstance(result, dict) else None
    if (not isinstance(pages, list) or not pages
            or any(not isinstance(page, dict) or not isinstance(page.get("markdown"), str) for page in pages)):
        raise RuntimeError("Mistral OCR returned no valid document pages.")
    return {
        "engine": "mistral", "file": str(path), "model": result.get("model", selected_model),
        "text": "\n\n".join(page["markdown"] for page in pages),
        "pages": pages, "usage_info": result.get("usage_info", {}),
    }

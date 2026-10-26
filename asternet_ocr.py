"""CPU ONNX adapter for asterxlabs/asternet-vietnamese-ocr.

The model recognizes text strips. Document mode uses RapidOCR only to detect
lines, then AsterNet recognizes each crop. See README_ASTERNET.md.
"""

import json
from pathlib import Path
import sys
import unicodedata


MODEL_ID = "asterxlabs/asternet-vietnamese-ocr"
MODEL_REVISION = "aec29ab0aaf26061e861cb4b3a301988744c5fa3"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / ".models" / "asternet"


def _model_files(model_dir=None):
    root = Path(model_dir).expanduser().resolve() if model_dir is not None else DEFAULT_MODEL_DIR
    names = ("asternet.onnx", "vocab.json", "config.json")
    missing = [name for name in names if not (root / name).is_file()]
    if model_dir is not None and missing:
        raise FileNotFoundError(f"AsterNet model folder is missing: {', '.join(missing)}")
    if missing:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {MODEL_ID} to {root}...", file=sys.stderr, flush=True)
        for name in missing:
            hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=str(root))
    return root


def preprocess(image, target_height=32):
    """Match the author's strip preprocessing, returning float32 NCHW in [0, 1].

    Reference: asterxlabs/asternet-vietnamese-ocr-demo,
    asterocr/inference/predictor.py (MIT).
    """
    import numpy as np
    from PIL import Image, ImageOps

    image = image.convert("RGB")
    width, height = image.size
    if height > width * 1.3:
        image = image.transpose(Image.Transpose.ROTATE_90)
        width, height = image.size
    gray_image = image.convert("L")
    gray = np.asarray(gray_image)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    center = gray[2:-2, 5:-5] if height > 4 and width > 10 else gray
    background = float(np.median(border))
    bright = float(np.percentile(center, 90)) - background
    dark = background - float(np.percentile(center, 10))
    if (bright > dark and bright > 20) or background < 90:
        image = ImageOps.invert(gray_image).convert("RGB")
    edge_color = tuple(int(v) for v in np.asarray(image)[:, [0, -1], :].mean(axis=(0, 1)))
    padded = Image.new("RGB", (width + 4, height), edge_color)
    padded.paste(image, (2, 0))
    aspect = padded.width / height
    scale = min(1.25, 1 + (aspect - 3) * .02) if aspect > 3 else 1.0
    new_width = max(16, int(target_height / height * padded.width * scale))
    new_width = ((new_width + 3) // 4) * 4
    if new_width > 8192:
        raise ValueError("Text strip is too wide; split it into shorter line crops.")
    resized = padded.resize((new_width, target_height), Image.Resampling.BILINEAR)
    tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(tensor[None])


def decode_ctc(log_probs, chars):
    """Greedy CTC decoding of upstream [time, batch=1, classes] log probabilities."""
    import numpy as np

    values = np.asarray(log_probs)
    if values.ndim != 3 or values.shape[1] != 1 or values.shape[2] != len(chars):
        raise RuntimeError(f"Unexpected AsterNet output shape: {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("AsterNet returned non-finite probabilities.")
    tokens = values[:, 0, :].argmax(axis=-1)
    previous = None
    text, scores = [], []
    for step, token in enumerate(tokens):
        token = int(token)
        if token != 0 and token != previous:
            text.append(chars[token])
            scores.append(float(np.exp(values[step, 0, token])))
        previous = token
    return {
        "text": unicodedata.normalize("NFC", "".join(text)).strip(),
        "confidence": sum(scores) / len(scores) if scores else 0.0,
    }


def _document_crops(image):
    """Return perspective-corrected crops in approximate row reading order."""
    import cv2
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    # RapidOCR expects BGR arrays. Its recognition/classification models are off.
    detector = RapidOCR(
        use_cls=False, use_rec=False, intra_op_num_threads=2, inter_op_num_threads=1,
    )
    boxes, _ = detector(np.asarray(image)[:, :, ::-1].copy(), use_cls=False, use_rec=False)
    if boxes is None or len(boxes) == 0:
        return []
    boxes = [np.asarray(box, dtype=np.float32) for box in boxes]
    boxes.sort(key=lambda box: float(box[:, 1].min()))
    rows = []
    for box in boxes:
        center = float(box[:, 1].mean())
        height = max(1.0, float(np.linalg.norm(box[0] - box[3])))
        for row in rows:
            if abs(center - row[0]) <= .5 * min(height, row[1]):
                row[2].append(box)
                break
        else:
            rows.append([center, height, [box]])
    source = np.asarray(image)
    crops = []
    for _, _, row_boxes in rows:
        for box in sorted(row_boxes, key=lambda box: float(box[:, 0].min())):
            width = max(1, round(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))))
            height = max(1, round(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))))
            if min(width, height) < 2:
                continue
            target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            matrix = cv2.getPerspectiveTransform(box, target)
            crop = cv2.warpPerspective(source, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
            crops.append((Image.fromarray(crop), box.tolist()))
    return crops


def extract_text(image_path: Path | str, *, mode="document", model_dir=None) -> dict:
    """Recognize an image locally using CPU; download missing model files once."""
    from PIL import Image, ImageOps

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if mode not in ("document", "line"):
        raise ValueError("AsterNet mode must be document or line.")
    if path.suffix.lower() == ".pdf":
        raise ValueError("AsterNet accepts images; convert PDF pages to images first.")
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) > 1:
            raise ValueError("Supply a single-frame image for AsterNet.")
        image = ImageOps.exif_transpose(source).convert("RGB")

    import onnxruntime as ort

    # Validate dependencies before downloading AsterNet's weights.
    if mode == "document":
        import rapidocr_onnxruntime  # noqa: F401
    root = _model_files(model_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    chars = json.loads((root / "vocab.json").read_text(encoding="utf-8"))["chars"]
    if not chars or chars[0] != "<blank>" or len(chars) != config["num_classes"]:
        raise ValueError("AsterNet vocabulary does not match its configuration.")
    print(f"Running AsterNet ({mode}) on CPU...", file=sys.stderr, flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / "asternet.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    crops = [(image, None)] if mode == "line" else _document_crops(image)
    lines = []
    for crop, polygon in crops:
        tensor = preprocess(crop, config["input_height"])
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        line = decode_ctc(output, chars)
        if line["text"]:
            if polygon is not None:
                line["polygon"] = polygon
            lines.append(line)
    return {
        "engine": "asternet", "model": MODEL_ID, "image": str(path),
        "language": "vi", "mode": mode, "device": "cpu", "decoder": "greedy_ctc",
        "original_size": list(image.size), "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
    }

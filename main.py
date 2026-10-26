"""Extract text with PaddleOCR, Tesseract, GLM-OCR, Mistral, Unlimited-OCR, or AsterNet.

Examples:
    python main.py "C:\\images\\id_card.jpg"
    python main.py "id_card.png" --output results
    python main.py "id_card.png" --raw --lang hi
    python main.py "id_card.png" --engine tesseract
    python main.py "id_card.png" --engine tesseract --lang eng+hin
    python main.py "id_card.png" --engine both
    python main.py "id_card.png" --engine glm
    python main.py "id_card.png" --engine mistral
    python main.py "id_card.png" --engine unlimited
    python main.py "image.png" --engine asternet
    python main.py "text_line.png" --engine asternet --asternet-mode line

Dependencies (CPU):
    python -m pip install "paddleocr>=3.0,<4" "paddlepaddle>=3.0,<4"
    python -m pip install pytesseract Pillow

Tesseract also requires the native Tesseract OCR program and language data.
GLM-OCR uses a separate glm_ocr.py module and local Ollama. Start Ollama and
run `ollama pull glm-ocr:q8_0` once. Its text/JSON is saved under output/glm.
Mistral uses mistral_ocr.py and requires MISTRAL_API_KEY. Its selected image or
PDF is sent to Mistral's hosted API; text/page JSON is saved under output/mistral.
Unlimited-OCR uses unlimited_ocr.py and a BF16-capable NVIDIA CUDA GPU.
See README_UNLIMITED.md for setup; text/JSON is saved under output/unlimited.
AsterNet uses asternet_ocr.py and CPU ONNX. See README_ASTERNET.md for setup;
Vietnamese text/JSON is saved under output/asternet.
PaddleOCR models are downloaded on first use. Image processing runs locally.
Large English/Hindi scans use the official mobile detector with a 960-pixel
detector limit. Smaller images retain the server detector for better accuracy.
PaddleOCR writes structured Aadhaar fields to <stem>.json, with raw recognition
in <stem>.ocr.json and <stem>.txt. Use --raw for the previous raw JSON format.
Unrecognized fields stay empty. City/district use comma-separated address order;
QR presence includes a dense-square texture fallback, without payload decoding.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import faulthandler
import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path


def extract_text(image_path: Path, lang: str = "en", structured: bool = False) -> dict:
    """Return recognized text and each line's confidence score."""
    from PIL import Image
    from paddleocr import PaddleOCR

    # Validate the local image before initializing or downloading models.
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()

    print("Loading PaddleOCR models...", file=sys.stderr, flush=True)
    # The server detector can exhaust memory or crash in native Windows CPU
    # inference on large scans. Use PaddleOCR's smaller official detector for
    # the English/Hindi workflows used here.
    use_mobile_detector = max(width, height) > 1280 and lang in ("en", "hi")
    detector_limit = 960 if use_mobile_detector else 1280
    model_options = (
        {
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": (
                "en_PP-OCRv5_mobile_rec" if lang == "en"
                else "devanagari_PP-OCRv5_mobile_rec"
            ),
        }
        if use_mobile_detector else {"lang": lang, "ocr_version": "PP-OCRv5"}
    )
    ocr = PaddleOCR(
        **model_options,
        device="cpu",
        cpu_threads=2,
        # Avoid a PaddlePaddle 3.3.x oneDNN/PIR compatibility error on Windows.
        enable_mkldnn=False,
        # Bound detector memory for large scans. Recognition still uses crops
        # from the original image, preserving detail in small printed text.
        text_det_limit_side_len=detector_limit,
        text_det_limit_type="max",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        # The line classifier can rotate clear upright Latin text by 180 degrees
        # on collages containing the front and back of a card. Whole-document
        # rotation is still enabled; vertical dates are read separately below.
        use_textline_orientation=False,
    )

    try:
        print(
            f"Running OCR on {width} x {height} pixels (detector maximum side: {detector_limit})...",
            file=sys.stderr, flush=True,
        )
        predictions = ocr.predict(input=str(image_path))
        print(
            f"OCR detected {sum(len(result['rec_texts']) for result in predictions)} text lines.",
            file=sys.stderr, flush=True,
        )
    finally:
        ocr.close()
    del ocr
    gc.collect()

    lines = []
    for result in predictions:
        page_lines = []
        for text, score, polygon in zip(
            result["rec_texts"], result["rec_scores"], result["rec_polys"]
        ):
            points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
            page_lines.append({
                "text": str(text), "confidence": float(score), "polygon": points,
            })
        lines.extend(page_lines)
        if structured:
            image = result["doc_preprocessor_res"]["output_img"]
            lines.extend(recognize_id_details(image, page_lines))

    return {
        "engine": "paddleocr",
        "image": str(image_path),
        "language": lang,
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
    }


def crop_line(image, polygon, margin=4):
    """Crop a detected line from the orientation-corrected document image."""
    height, width = image.shape[:2]
    xs, ys = zip(*polygon)
    return image[
        max(0, int(min(ys)) - margin):min(height, int(max(ys)) + margin + 1),
        max(0, int(min(xs)) - margin):min(width, int(max(xs)) + margin + 1),
    ].copy()


def recognize_id_details(image, lines):
    """Read vertical dates and the Hindi line above the English name."""
    import cv2
    from paddleocr import TextRecognition

    extra = []
    vertical = []
    vertical_boxes = []
    for line in lines:
        xs, ys = zip(*line["polygon"])
        if max(ys) - min(ys) > 2.5 * max(1, max(xs) - min(xs)):
            vertical_boxes.append([min(xs), min(ys), max(xs), max(ys)])
    # Detection may split a vertical date into separate label and digit boxes.
    merged = []
    for box in sorted(vertical_boxes, key=lambda box: box[1]):
        for group in merged:
            width = max(group[2] - group[0], box[2] - box[0])
            if (abs((box[0] + box[2] - group[0] - group[2]) / 2) <= width
                    and 0 <= box[1] - group[3] <= 6 * width):
                group[:] = [min(group[0], box[0]), min(group[1], box[1]),
                            max(group[2], box[2]), max(group[3], box[3])]
                break
        else:
            merged.append(box.copy())
    for x1, y1, x2, y2 in merged:
        crop = crop_line(image, [[x1, y1], [x2, y2]])
        vertical.extend([
            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ])
    if vertical:
        print("Reading sideways issue/download dates...", file=sys.stderr)
        model = TextRecognition(
            model_name="en_PP-OCRv5_mobile_rec", device="cpu", enable_mkldnn=False, cpu_threads=2,
        )
        for prediction in model.predict(vertical, batch_size=1):
            extra.append({
                "text": str(prediction["rec_text"]),
                "confidence": float(prediction["rec_score"]), "source": "vertical",
            })
        model.close()

    name_line = find_name_line(lines)
    if name_line and "polygon" in name_line:
        name_x = min(point[0] for point in name_line["polygon"])
        name_top = min(point[1] for point in name_line["polygon"])
        name_height = max(point[1] for point in name_line["polygon"]) - name_top
        name_center = name_top + name_height / 2
        candidates = []
        for line in lines:
            xs, ys = zip(*line["polygon"])
            if (0 < name_center - (min(ys) + max(ys)) / 2 < 2.5 * name_height
                    and abs(min(xs) - name_x) < 2 * name_height):
                candidates.append(line)
        if candidates:
            closest = max(candidates, key=lambda line: sum(p[1] for p in line["polygon"]) / len(line["polygon"]))
            print("Loading Hindi name recognizer (downloads on first use)...", file=sys.stderr)
            model = TextRecognition(
                model_name="devanagari_PP-OCRv5_mobile_rec",
                device="cpu", enable_mkldnn=False, cpu_threads=2,
            )
            crop = crop_line(image, closest["polygon"])
            for prediction in model.predict(crop):
                extra.append({
                    "text": str(prediction["rec_text"]),
                    "confidence": float(prediction["rec_score"]), "source": "hindi_name",
                })
            model.close()
    return extra


DOB_LABEL = (
    r"\bDOB|date\s*of\s*birth|जन्म|"
    r"(?:^|[/\s])D[O0B]?B?(?=\s*[:\-]?\s*\d{1,2}\s*[/.-]\s*\d{1,2})"
)


def find_name_line(lines):
    """Prefer a name label, otherwise the nearest English line before DOB."""
    for line in lines:
        if re.match(r"^name\s*[:\-]", line["text"], re.I):
            return line
    for index, line in enumerate(lines):
        if re.search(DOB_LABEL, line["text"], re.I):
            for candidate in reversed(lines[max(0, index - 3):index]):
                text = candidate["text"].strip()
                if (candidate.get("confidence", 1) >= 0.8
                        and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,79}", text)
                        and not re.search(r"government|india|aadhaar|female|male|birth|^DOB$", text, re.I)):
                    return candidate
    return None


def is_aadhaar_document(lines):
    """Use independent card signals when OCR damages the Aadhaar heading."""
    text = "\n".join(line["text"] for line in lines)
    if re.search(
        r"aadha[ar]+|uidai|u?idai[\s.]*gov[\s.]*in|"
        r"unique\s+identification\s+authority|आधार", text, re.I,
    ):
        return True
    numbers = []
    for line in lines:
        if line.get("confidence", 1) < .8:
            continue
        number = re.fullmatch(r"\s*([2-9]\d{3}[ -]?\d{4}[ -]?\d{4})\s*", line["text"])
        if number:
            numbers.append(re.sub(r"\D", "", number[1]))
    repeated_number = any(numbers.count(number) >= 2 for number in set(numbers))
    has_address = bool(re.search(r"\baddress\s*:", text, re.I))
    has_gender = bool(re.search(r"\b(?:female|male|transgender)\b|महिला|पुरुष", text, re.I))
    has_birth = bool(re.search(DOB_LABEL, text, re.I))
    has_government = bool(re.search(r"government\s+of\s+india", text, re.I))
    return bool(
        (repeated_number and has_address and has_gender)
        or (numbers and has_government and has_birth and has_gender)
    )


STATES = (
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam",
    "Bihar", "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
)


def parse_aadhaar(raw, qr_present=False):
    """Extract supported Aadhaar fields; missing or ambiguous values stay empty."""
    document = {
        "document_type": "", "name": "", "name_hindi": "", "date_of_birth": "",
        "gender": "", "aadhaar_number": "", "virtual_id": "", "issue_date": "",
        "download_date": "",
        "address": dict.fromkeys((
            "care_of", "house_number", "locality", "city", "district", "state",
            "pin_code", "country",
        ), ""),
        "issuing_authority": "", "qr_code_present": bool(qr_present),
    }
    lines = raw["lines"]
    text = "\n".join(line["text"] for line in lines)
    if is_aadhaar_document(lines):
        document["document_type"] = "Aadhaar"
        document["issuing_authority"] = "UIDAI"
    else:
        return document

    name_line = find_name_line(lines)
    if name_line:
        document["name"] = re.sub(r"^name\s*[:\-]\s*", "", name_line["text"], flags=re.I).strip()
    hindi = [line for line in lines if line.get("source") == "hindi_name"
             and line.get("confidence", 0) >= 0.7 and re.search(r"[\u0900-\u097f]", line["text"])]
    if hindi:
        document["name_hindi"] = max(hindi, key=lambda line: line["confidence"])["text"].strip()

    for line in lines:
        value = line["text"]
        if line.get("confidence", 1) < 0.7:
            continue
        # Parse dates only with a matching label; never derive full DOB from a year.
        date = re.search(r"(?<!\d)(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})(?!\d)", value)
        if date:
            try:
                iso_date = datetime(int(date[3]), int(date[2]), int(date[1])).date().isoformat()
            except ValueError:
                iso_date = ""
            for key, label in (
                ("date_of_birth", DOB_LABEL),
                ("issue_date", r"issue\s*date|issued"),
                ("download_date", r"download\s*date|downloaded"),
            ):
                if re.search(label, value, re.I):
                    document[key] = iso_date
        if re.search(r"\bFEMALE\b|महिला", value, re.I):
            document["gender"] = "Female"
        elif re.search(r"\bMALE\b|पुरुष", value, re.I):
            document["gender"] = "Male"
        elif re.search(r"\bTRANSGENDER\b", value, re.I):
            document["gender"] = "Transgender"
        if re.search(r"\bVID\b|virtual\s*id", value, re.I):
            number = re.search(r"(?<!\d)(?:\d[ -]*){15}\d(?![ -]*\d)", value)
            if number:
                document["virtual_id"] = re.sub(r"\D", "", number[0])
        else:
            # Extract the printed digits, including demonstration card numbers;
            # this function does not validate an Aadhaar number's authenticity.
            number = re.fullmatch(r"\s*(?:aadhaar(?:\s*(?:no\.?|number))?\s*[:\-]?\s*)?(\d{4}[ -]?\d{4}[ -]?\d{4})\s*", value, re.I)
            if number:
                document["aadhaar_number"] = re.sub(r"\D", "", number[1])

    address = document["address"]
    start = next((index for index, line in enumerate(lines)
                  if re.match(r"^address\s*[:\-]?", line["text"], re.I)), None)
    if start is not None:
        address_lines = []
        first = re.sub(r"^address\s*[:\-]?\s*", "", lines[start]["text"], flags=re.I)
        for value in [first] + [line["text"] for line in lines[start + 1:]]:
            if re.search(r"\bVID\b|www\.|@|^\s*\d{4}\s+\d{4}\s+\d{4}\s*$", value, re.I):
                break
            # Skip short noise and Hindi when parsing the English address.
            if len(value.strip()) < 2 or re.search(r"[\u0900-\u097f]", value):
                continue
            address_lines.append(value.strip())
            if re.search(r"\b[1-9]\d{5}\b", value):
                break
        # OCR line wrapping is not an address separator. Join wrapped words
        # before interpreting the printed commas.
        parts = [part.strip() for part in " ".join(address_lines).split(",") if part.strip()]
        remaining = []
        for part in parts:
            care = re.match(r"(?:C\s*/\s*O|S\s*/\s*O|D\s*/\s*O|W\s*/\s*O|care\s+of)\s*[:\-]?\s*(.+)", part, re.I)
            if care:
                address["care_of"] = care[1].strip().title()
                continue
            pin = re.search(r"\b[1-9]\d{5}\b", part)
            if pin:
                address["pin_code"] = pin[0]
            state = next((state for state in STATES if re.search(r"\b" + re.escape(state) + r"\b", part, re.I)), None)
            if state:
                address["state"] = state
                continue
            if pin:
                part = part[:pin.start()].strip(" -:")
            if part:
                remaining.append(part)
        if remaining and re.fullmatch(r"(?:house\s*(?:no\.?|number)?\s*[:\-]?\s*)?[\dA-Za-z/-]*\d[\dA-Za-z/-]*", remaining[0], re.I):
            address["house_number"] = re.sub(r"^house\s*(?:no\.?|number)?\s*[:\-]?\s*", "", remaining.pop(0), flags=re.I)
        # Unlabelled comma-separated addresses conventionally end in city, district.
        if len(remaining) >= 3:
            address["locality"] = ", ".join(remaining[:-2]).title()
            address["city"], address["district"] = (part.title() for part in remaining[-2:])
        elif len(remaining) == 2:
            address["locality"], address["city"] = (part.title() for part in remaining)
        elif remaining:
            address["locality"] = remaining[0].title()
        # G.P.O./H.O./S.O./B.O. describe the post office, not part of the city name.
        address["city"] = re.sub(
            r"\s+(?:G\.?\s*P\.?\s*O\.?|H\.?\s*O\.?|S\.?\s*O\.?|B\.?\s*O\.?)\s*$",
            "", address["city"], flags=re.I,
        )
    # Enrolment letters often omit "Address:" and use separate labelled rows.
    for line in lines:
        if line.get("confidence", 1) < .7:
            continue
        value = line["text"].strip()
        labels = (
            ("care_of", r"(?:C\s*/\s*O|S\s*/\s*O|D\s*/\s*O|W\s*/\s*O)"),
            ("district", r"District"), ("state", r"State"),
            ("pin_code", r"PIN\s*(?:Code)?"), ("city", r"City"),
        )
        for field, label in labels:
            match = re.match(r"^(?:" + label + r")\s*[:\-]\s*(.+)", value, re.I)
            if not match:
                continue
            content = match[1].split(",")[0].strip().rstrip(".")
            if field == "pin_code":
                pin = re.search(r"\b[1-9]\d{5}\b", content)
                if pin:
                    address[field] = pin[0]
            elif content:
                address[field] = content.title()
        post_office = re.match(r"^P\.?\s*O\.?\s*:\s*(.+)", value, re.I)
        if post_office and not address["city"]:
            address["city"] = post_office[1].strip(" ,.").title()
        locality = re.match(r"^VTC\s*:\s*(.+)", value, re.I)
        if locality:
            house = re.fullmatch(
                r"(?:Flat|House)\s*(?:No\.?|Number)?\s*[:\-]?\s*([\w/-]+)",
                locality[1].strip(" ,."), re.I,
            )
            if house:
                address["house_number"] = house[1]
            elif not address["locality"]:
                address["locality"] = locality[1].strip(" ,.").title()
    if address["state"] or address["pin_code"]:
        address["country"] = "India"
    return document


def detect_qr_code(image_path):
    """Detect QR geometry or a dense square code region, without reading payloads.

    The texture fallback handles dense/blurred printed Aadhaar codes for which
    OpenCV cannot recover QR finder geometry. It is a presence heuristic.
    """
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    detector = cv2.QRCodeDetector()
    detected, _ = detector.detect(image)
    if detected:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if not (0.75 < width / height < 1.3 and 0.03 * gray.size < area < 0.4 * gray.size):
            continue
        edges = cv2.Canny(gray[y:y + height, x:x + width], 70, 160)
        if (edges > 0).mean() > 0.25:
            return True
    return False


def extract_tesseract_text(
    image_path: Path, lang: str = "en", tesseract_cmd: str | None = None
) -> dict:
    """Group Tesseract words into lines, with mean word confidence (0 to 1)."""
    from PIL import Image
    import pytesseract

    executable = tesseract_cmd or os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not executable:
        candidates = [Path(__file__).parent / ".tools" / "Tesseract-OCR" / "tesseract.exe"]
        if os.name == "nt":
            for root in (
                os.environ.get("ProgramFiles", "C:/Program Files"),
                os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", ""),
            ):
                candidates.extend([
                    Path(root) / "Tesseract-OCR" / "tesseract.exe",
                    Path(root) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
                ])
        executable = next((str(path) for path in candidates if path.is_file()), None)
    if not executable:
        raise RuntimeError(
            "Tesseract OCR executable not found. Install Tesseract OCR and add it "
            "to PATH, or use --tesseract-cmd with the full path to tesseract.exe."
        )
    pytesseract.pytesseract.tesseract_cmd = executable

    # Accept the existing PaddleOCR codes as well as native Tesseract codes.
    language = "+".join({"en": "eng", "hi": "hin"}.get(code, code) for code in lang.split("+"))
    available = pytesseract.get_languages(config="")
    missing = set(language.split("+")) - set(available)
    if missing:
        raise RuntimeError(
            f"Tesseract language data not installed: {', '.join(sorted(missing))}. "
            f"Installed languages: {', '.join(available)}. "
            "Install the matching .traineddata files in Tesseract's tessdata folder."
        )
    with Image.open(image_path) as image:
        # Convert WebP and other input formats to an image Tesseract can read.
        data = pytesseract.image_to_data(
            image.convert("RGB"), lang=language, output_type=pytesseract.Output.DICT
        )

    grouped = {}
    for index, word in enumerate(data["text"]):
        word = word.strip()
        confidence = float(data["conf"][index])
        if not word or confidence < 0:
            continue
        key = tuple(data[field][index] for field in ("page_num", "block_num", "par_num", "line_num"))
        line = grouped.setdefault(key, {"words": [], "scores": []})
        line["words"].append(word)
        line["scores"].append(confidence / 100)
    lines = [
        {"text": " ".join(line["words"]), "confidence": sum(line["scores"]) / len(line["scores"])}
        for line in grouped.values()
    ]
    return {
        "engine": "tesseract",
        "image": str(image_path),
        "language": language,
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text using PaddleOCR, Tesseract, GLM-OCR, Mistral, Unlimited-OCR, or AsterNet."
    )
    parser.add_argument("image", type=Path, help="Path to a local image (Mistral also accepts PDF)")
    parser.add_argument(
        "--engine", choices=("paddleocr", "tesseract", "glm", "glm-ocr", "mistral", "unlimited", "unlimited-ocr", "asternet", "both"), default="paddleocr",
        help="OCR engine (default: paddleocr)",
    )
    parser.add_argument(
        "--lang", default="en",
        help="Language for raw OCR/Tesseract (default: en; Hindi: hi; Tesseract: eng+hin). Structured PaddleOCR uses English and Hindi.",
    )
    parser.add_argument("--tesseract-cmd", help="Full path to the Tesseract executable, if needed")
    parser.add_argument("--glm-model", help="Ollama GLM model (default: glm-ocr:q8_0)")
    parser.add_argument("--glm-host", help="Ollama server URL (default: OLLAMA_HOST or http://127.0.0.1:11434)")
    parser.add_argument("--glm-timeout", type=float, default=600, help="GLM request timeout in seconds (default: 600)")
    parser.add_argument("--mistral-model", help="Mistral model (default: mistral-ocr-latest)")
    parser.add_argument("--unlimited-model", default="baidu/Unlimited-OCR", help="Unlimited OCR Hugging Face model ID or local model folder")
    parser.add_argument("--unlimited-mode", choices=("gundam", "base"), default="gundam", help="Unlimited OCR image mode (default: gundam)")
    parser.add_argument("--unlimited-max-length", type=int, default=32768, help="Unlimited OCR total token limit, including input (1-32768)")
    parser.add_argument("--asternet-mode", choices=("document", "line"), default="document", help="AsterNet: detect document lines or recognize one cropped line (default: document)")
    parser.add_argument("--asternet-model-dir", type=Path, help="Local AsterNet folder containing asternet.onnx, vocab.json, and config.json")
    parser.add_argument(
        "--raw", action="store_true", help="Save only raw OCR instead of structured PaddleOCR JSON",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output"), help="Output folder (default: output)"
    )
    args = parser.parse_args()
    if args.engine == "glm-ocr":
        args.engine = "glm"
    if args.engine == "unlimited-ocr":
        args.engine = "unlimited"
    if not 1 <= args.unlimited_max_length <= 32768:
        parser.error("--unlimited-max-length must be between 1 and 32768.")
    if args.glm_timeout <= 0:
        parser.error("--glm-timeout must be greater than zero.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    image_path = args.image.expanduser().resolve()

    if not image_path.is_file():
        parser.error(f"Image file not found: {image_path}")
    if image_path.suffix.lower() == ".pdf" and args.engine != "mistral":
        parser.error("PDF input currently requires --engine mistral.")

    engines = ("paddleocr", "tesseract") if args.engine == "both" else (args.engine,)
    failed = False
    for engine in engines:
        try:
            if engine == "asternet":
                from asternet_ocr import extract_text as extract_asternet_text

                result = extract_asternet_text(
                    image_path, mode=args.asternet_mode, model_dir=args.asternet_model_dir,
                )
            elif engine == "unlimited":
                from unlimited_ocr import extract_text as extract_unlimited_text

                result = extract_unlimited_text(
                    image_path, model=args.unlimited_model, mode=args.unlimited_mode,
                    max_length=args.unlimited_max_length,
                )
            elif engine == "mistral":
                from mistral_ocr import extract_text as extract_mistral_text

                print("Sending document to Mistral OCR...", file=sys.stderr, flush=True)
                result = extract_mistral_text(image_path, model=args.mistral_model)
            elif engine == "glm":
                from glm_ocr import extract_text as extract_glm_text

                print("Running GLM-OCR through Ollama...", file=sys.stderr, flush=True)
                result = extract_glm_text(
                    image_path, model=args.glm_model, host=args.glm_host, timeout=args.glm_timeout,
                )
            elif engine == "tesseract":
                result = extract_tesseract_text(image_path, args.lang, args.tesseract_cmd)
            else:
                with redirect_stdout(sys.stderr):
                    # Structured extraction uses English for Latin fields and a
                    # targeted Hindi recognizer for the name above the English name.
                    result = extract_text(image_path, args.lang if args.raw else "en", structured=not args.raw)
            # Preserve the original PaddleOCR output paths; keep Tesseract separate.
            output_dir = args.output if engine == "paddleocr" else args.output / engine
            output_dir.mkdir(parents=True, exist_ok=True)
            text_path = output_dir / f"{image_path.stem}.txt"
            json_path = output_dir / f"{image_path.stem}.json"
            text_path.write_text(result["text"], encoding="utf-8")
            structured_result = None
            if engine == "paddleocr" and not args.raw:
                structured_result = parse_aadhaar(result, detect_qr_code(image_path))
                raw_path = output_dir / f"{image_path.stem}.ocr.json"
                raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            json_path.write_text(
                json.dumps(structured_result if structured_result is not None else result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except ImportError as exc:
            packages = (
                "-r requirements-asternet.txt" if engine == "asternet"
                else "-r requirements-unlimited.txt (also install CUDA PyTorch; see README_UNLIMITED.md)" if engine == "unlimited"
                else "Pillow" if engine in ("glm", "mistral") else "pytesseract Pillow" if engine == "tesseract"
                else '"paddleocr>=3.0,<4" "paddlepaddle>=3.0,<4"'
            )
            print(f"Missing dependency: {exc}\nInstall: python -m pip install {packages}", file=sys.stderr)
            failed = True
            continue
        except Exception as exc:
            print(f"{engine} OCR failed: {exc}", file=sys.stderr)
            failed = True
            continue

        if structured_result is not None:
            print(json.dumps(structured_result, ensure_ascii=False, indent=2))
        else:
            print(f"\n[{engine}]")
            print(result["text"] or "No text detected. Try a clearer image or a different --lang.")
        print(f"\nText saved to: {text_path.resolve()}", file=sys.stderr)
        print(f"JSON saved to: {json_path.resolve()}", file=sys.stderr)
        if result.get("truncated"):
            print("GLM reached its generation limit; the saved text is incomplete. Try smaller image sections.", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    # Native Paddle failures bypass Python exceptions; show diagnostics instead
    # of silently returning to the PowerShell prompt.
    faulthandler.enable()
    raise SystemExit(main())

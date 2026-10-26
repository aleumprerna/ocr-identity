# AsterNet Vietnamese OCR

`main.py` calls `asternet_ocr.extract_text()` for `--engine asternet`.
This adapter runs the published ONNX model on CPU, with greedy CTC decoding.
No API key, Ollama, PyTorch, or GPU is required.

## Install and test (PowerShell)

```powershell
cd C:\OCR-New
python -m venv .venv-asternet
.\.venv-asternet\Scripts\python.exe -m pip install -r requirements-asternet.txt

# Whole image: detect text lines, recognize each with AsterNet.
.\.venv-asternet\Scripts\python.exe main.py image.png --engine asternet

# One tightly cropped horizontal text line: skip the line detector.
.\.venv-asternet\Scripts\python.exe main.py text_line.png --engine asternet --asternet-mode line

# Automated checks, without downloading or loading models.
.\.venv-asternet\Scripts\python.exe -m unittest test_asternet_ocr -v
```

Replace `image.png` with your image path. Quote paths containing spaces.
The repository already contains `image.png`, `adhar.webp`, `pan.png`, and other
images for trying document mode. They do not provide a Vietnamese accuracy benchmark.

Results are printed and saved as `output\asternet\<image-name>.txt` and `.json`.
Use `--output results` to save under `results\asternet` instead.
JSON includes raw text, lines, confidence, and detected polygons in document mode.
Confidence is the mean greedy character probability, not a calibrated accuracy score.

The first run downloads approximately 41 MB of AsterNet weights plus its vocabulary
and configuration into `.models\asternet`. The download is pinned to model commit
`aec29ab0aaf26061e861cb4b3a301988744c5fa3`. Later runs reuse these files locally.
RapidOCR supplies a separate bundled line detector; its recognizer is disabled.
Inference stays on your machine.

To use existing model files without downloading AsterNet:

```powershell
.\.venv-asternet\Scripts\python.exe main.py image.png --engine asternet --asternet-model-dir .models\asternet
```

That folder must contain `asternet.onnx`, `config.json`, and `vocab.json`.
The adapter recognizes images only; convert PDF pages into images first.

## Scope

AsterNet is a Vietnamese/Latin text-line recognizer. Hindi/Devanagari is absent
from its published vocabulary, so use another engine for Hindi text. The adapter
does not extract structured Aadhaar fields. Document reading order is approximate
top to bottom, then left to right, and may need adjustment for complex columns.
It follows the author's image preprocessing but uses greedy CTC directly, without
the separate linguistic rescorer or beam search; published benchmark scores do
not establish the accuracy of this adapter.

Sources: [model and artifacts](https://huggingface.co/asterxlabs/asternet-vietnamese-ocr),
[author's preprocessing](https://huggingface.co/spaces/asterxlabs/asternet-vietnamese-ocr-demo/blob/main/asterocr/inference/predictor.py).

# Baidu Unlimited-OCR

`unlimited_ocr.py` implements local image OCR with `baidu/Unlimited-OCR`.
`main.py` imports it when you select `--engine unlimited` (alias: `unlimited-ocr`).

Baidu documents NVIDIA GPU inference and a Python 3.12 / CUDA 12.9 setup.
This adapter requires a CUDA GPU with BF16 support. CPU-only PyTorch cannot run it.
The first run downloads model weights and custom Python code from Hugging Face
using `trust_remote_code=True`; image inference then runs locally.

## Setup in PowerShell

Use a separate environment for this engine. Python 3.12 and a compatible NVIDIA
driver must be installed first.

```powershell
cd C:\OCR-New
py -3.12 -m venv .venv-unlimited
.\.venv-unlimited\Scripts\python.exe -m pip install --upgrade pip
.\.venv-unlimited\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129
.\.venv-unlimited\Scripts\python.exe -m pip install -r requirements-unlimited.txt
```

Check GPU access:

```powershell
.\.venv-unlimited\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Run

```powershell
.\.venv-unlimited\Scripts\python.exe main.py "pan.png" --engine unlimited
.\.venv-unlimited\Scripts\python.exe main.py "adhar.webp" --engine unlimited --unlimited-mode base
```

The first command saves `output/unlimited/pan.txt` and
`output/unlimited/pan.json`. Use `--output results` to change the parent folder.
The JSON contains the model ID, image path, raw recognized text, and settings.
The text preserves any layout tags emitted by the model.
This adapter accepts one image at a time; convert PDF pages to images first.
The upstream inference method returns text without a completion reason, so this
adapter cannot identify output cut off at the token limit. The default limit is
32768 total tokens; `--unlimited-max-length` can lower it.

From Python:

```python
from unlimited_ocr import extract_text

result = extract_text("pan.png")
print(result["text"])
```

References: [Baidu model card](https://huggingface.co/baidu/Unlimited-OCR),
[Baidu repository](https://github.com/baidu/Unlimited-OCR).

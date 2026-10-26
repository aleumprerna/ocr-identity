# GLM-OCR

`main.py` calls `glm_ocr.extract_text()` when you choose `--engine glm` (or
`--engine glm-ocr`). The module uses the official GLM-OCR model through Ollama.
It returns raw text, which can contain Markdown, and generation metadata.
PaddleOCR remains the default engine for structured Aadhaar JSON.

Start the Ollama application, then download the model once:

```powershell
ollama pull glm-ocr:q8_0
```

Extract text:

```powershell
cd C:\OCR-New
python main.py image.png --engine glm
```

Results are saved in `output\glm\image.txt` and `output\glm\image.json`.
No API key or additional Python SDK is needed. The default endpoint is local.
The model download requires internet access and disk space; local inference
requires available memory and may be slower on CPU.

For an existing Ollama deployment:

```powershell
python main.py image.png --engine glm --glm-model glm-ocr:q8_0 --glm-host http://127.0.0.1:11434 --glm-timeout 900
```

`GLM_OCR_MODEL` and `OLLAMA_HOST` can also configure the model and server.
If you select a remote server, the image is sent to that server.

Run checks without downloading or loading a model:

```powershell
python -m unittest test_main test_glm_ocr -v
```

References: [GLM-OCR](https://github.com/zai-org/GLM-OCR),
[Ollama model](https://registry.ollama.com/library/glm-ocr),
[vision API](https://docs.ollama.com/capabilities/vision).

The previously requested Mistral option is implemented separately in
`mistral_ocr.py`. It uses Mistral's hosted API and requires a key:

```powershell
$env:MISTRAL_API_KEY = "your_api_key"
python main.py image.png --engine mistral
python main.py document.pdf --engine mistral
```

The selected file is sent to Mistral, and text/page JSON is saved under
`output\mistral`. Its API checks use mocked responses until a key is configured.
Run every test with `python -m unittest discover -v`.

"""GLM adapter and main.py integration checks without model downloads."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from PIL import Image

import glm_ocr
import main


class GLMOCRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.image = self.folder / "sample.webp"
        Image.new("RGB", (100, 50), "white").save(self.image)

    def response(self, **overrides):
        result = {
            "model": "glm-ocr:q8_0", "done": True, "done_reason": "stop",
            "message": {"content": "Demo Person\nनमस्ते\nDOB: 01/01/2000"},
            "eval_count": 20,
        }
        result.update(overrides)
        return io.BytesIO(json.dumps(result, ensure_ascii=False).encode("utf-8"))

    def test_image_and_unicode_response(self):
        with patch("glm_ocr.urlopen", return_value=self.response()) as send:
            result = glm_ocr.extract_text(self.image, host="http://127.0.0.1:11434")
        self.assertIn("नमस्ते", result["text"])
        self.assertFalse(result["truncated"])
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(payload["messages"][0]["content"], "Text Recognition:")
        self.assertTrue(payload["messages"][0]["images"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], 0)

    def test_model_not_downloaded_has_setup_command(self):
        error = HTTPError("http://127.0.0.1:11434/api/chat", 404, "not found", {}, None)
        with patch("glm_ocr.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "ollama pull glm-ocr:q8_0"):
                glm_ocr.extract_text(self.image, model="glm-ocr:q8_0")

    def test_server_not_running_has_actionable_error(self):
        with patch("glm_ocr.urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(RuntimeError, "ollama serve"):
                glm_ocr.extract_text(self.image)

    def test_empty_or_incomplete_response_is_not_success(self):
        for response in ({"message": {"content": ""}}, {"done": False}):
            with self.subTest(response=response):
                with patch("glm_ocr.urlopen", return_value=self.response(**response)):
                    with self.assertRaises(RuntimeError):
                        glm_ocr.extract_text(self.image)

    def test_generation_limit_is_reported(self):
        with patch("glm_ocr.urlopen", return_value=self.response(done_reason="length")):
            result = glm_ocr.extract_text(self.image)
        self.assertTrue(result["truncated"])

    def test_large_input_is_bounded(self):
        Image.new("RGB", (3000, 2000), "white").save(self.image)
        with patch("glm_ocr.urlopen", return_value=self.response()):
            result = glm_ocr.extract_text(self.image)
        self.assertEqual(result["original_size"], [3000, 2000])
        self.assertEqual(max(result["input_size"]), 1600)

    def test_main_calls_separate_module_and_writes_glm_outputs(self):
        argv = ["main.py", str(self.image), "--engine", "glm-ocr", "--output", str(self.folder)]
        with patch("sys.argv", argv), patch("glm_ocr.urlopen", return_value=self.response()), \
                patch("main.extract_text", side_effect=AssertionError("PaddleOCR must not run")), \
                redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            status = main.main()
        self.assertEqual(status, 0)
        output = self.folder / "glm"
        text = (output / "sample.txt").read_text(encoding="utf-8")
        result = json.loads((output / "sample.json").read_text(encoding="utf-8"))
        self.assertEqual(result["engine"], "glm")
        self.assertEqual(result["text"], text)
        self.assertIn("Demo Person", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

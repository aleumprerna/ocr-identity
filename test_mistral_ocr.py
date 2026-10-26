"""Mistral adapter checks using mocked hosted API responses."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import main
import mistral_ocr


class MistralOCRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.pdf = self.folder / "sample.pdf"
        self.pdf.write_bytes(b"%PDF-1.4\n%%EOF")

    def response(self):
        return io.BytesIO(json.dumps({
            "model": "mistral-ocr-latest",
            "pages": [{"index": 0, "markdown": "Page one"}, {"index": 1, "markdown": "नमस्ते"}],
        }, ensure_ascii=False).encode("utf-8"))

    def test_pdf_and_multiple_pages(self):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}), \
                patch("mistral_ocr.urlopen", return_value=self.response()) as send:
            result = mistral_ocr.extract_text(self.pdf)
        self.assertEqual(result["text"], "Page one\n\nनमस्ते")
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["document"]["type"], "document_url")
        self.assertTrue(payload["document"]["document_url"].startswith("data:application/pdf;base64,"))

    def test_missing_api_key_does_not_send_a_file(self):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": ""}), patch("mistral_ocr.urlopen") as send:
            with self.assertRaisesRegex(RuntimeError, "MISTRAL_API_KEY"):
                mistral_ocr.extract_text(self.pdf)
            send.assert_not_called()

    def test_authentication_error_is_actionable(self):
        error = HTTPError("https://api.mistral.ai/v1/ocr", 401, "Unauthorized", {}, None)
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}), \
                patch("mistral_ocr.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401.*MISTRAL_API_KEY"):
                mistral_ocr.extract_text(self.pdf)

    def test_main_routes_pdf_to_mistral(self):
        argv = ["main.py", str(self.pdf), "--engine", "mistral", "--output", str(self.folder)]
        with patch("sys.argv", argv), patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}), \
                patch("mistral_ocr.urlopen", return_value=self.response()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main.main(), 0)
        result = json.loads((self.folder / "mistral/sample.json").read_text(encoding="utf-8"))
        self.assertEqual(result["engine"], "mistral")
        self.assertEqual(result["text"], (self.folder / "mistral/sample.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

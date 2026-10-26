"""Unlimited OCR contract and CLI checks without GPU/model downloads."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import main
import unlimited_ocr


class UnlimitedOCRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.image = self.folder / "sample.webp"
        Image.new("RGB", (100, 50), "white").save(self.image)
        self.ocr = MagicMock()
        self.ocr.infer.return_value = "  Demo Person\nनमस्ते  "
        self.loader = MagicMock()
        self.loader.from_pretrained.return_value.eval.return_value.cuda.return_value = self.ocr
        self.tokenizer = MagicMock()
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
            bfloat16="bfloat16", inference_mode=nullcontext,
        )
        self.modules = {
            "torch": self.torch,
            "transformers": SimpleNamespace(AutoModel=self.loader, AutoTokenizer=self.tokenizer),
        }

    def test_modes_return_model_text_and_clean_scratch(self):
        for mode, size, crop in (("gundam", 640, True), ("base", 1024, False)):
            with self.subTest(mode=mode), patch.dict("sys.modules", self.modules), redirect_stderr(io.StringIO()):
                result = unlimited_ocr.extract_text(self.image, mode=mode)
            self.assertEqual(result["text"], "Demo Person\nनमस्ते")
            self.assertEqual(result["model"], "baidu/Unlimited-OCR")
            args = self.ocr.infer.call_args.kwargs
            self.assertEqual(args["image_size"], size)
            self.assertEqual(args["crop_mode"], crop)
            self.assertTrue(args["eval_mode"])
            self.assertFalse(Path(args["output_path"]).exists())

    def test_no_cuda_fails_before_downloading_model(self):
        self.torch.cuda.is_available = lambda: False
        with patch.dict("sys.modules", self.modules):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                unlimited_ocr.extract_text(self.image)
        self.loader.from_pretrained.assert_not_called()
        self.tokenizer.from_pretrained.assert_not_called()

    def test_bf16_required_before_download(self):
        self.torch.cuda.is_bf16_supported = lambda: False
        with patch.dict("sys.modules", self.modules):
            with self.assertRaisesRegex(RuntimeError, "BF16"):
                unlimited_ocr.extract_text(self.image)
        self.loader.from_pretrained.assert_not_called()

    def test_empty_response_is_failure(self):
        self.ocr.infer.return_value = None
        with patch.dict("sys.modules", self.modules), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "returned no text"):
                unlimited_ocr.extract_text(self.image)

    def test_main_writes_unicode_outputs_and_forwards_options(self):
        argv = ["main.py", str(self.image), "--engine", "unlimited-ocr",
                "--unlimited-mode", "base", "--unlimited-max-length", "4096",
                "--output", str(self.folder)]
        with patch("sys.argv", argv), patch.dict("sys.modules", self.modules), \
                patch("main.extract_text", side_effect=AssertionError("PaddleOCR must not run")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main.main(), 0)
        result = json.loads((self.folder / "unlimited/sample.json").read_text(encoding="utf-8"))
        self.assertEqual(result["text"], (self.folder / "unlimited/sample.txt").read_text(encoding="utf-8"))
        self.assertEqual(result["mode"], "base")
        self.assertEqual(self.ocr.infer.call_args.kwargs["max_length"], 4096)

    def test_main_returns_failure_without_writing_outputs(self):
        self.torch.cuda.is_available = lambda: False
        argv = ["main.py", str(self.image), "--engine", "unlimited", "--output", str(self.folder)]
        with patch("sys.argv", argv), patch.dict("sys.modules", self.modules), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main.main(), 1)
        self.assertIn("CUDA", stderr.getvalue())
        self.assertFalse((self.folder / "unlimited").exists())


if __name__ == "__main__":
    unittest.main()

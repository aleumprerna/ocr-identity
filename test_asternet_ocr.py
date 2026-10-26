"""AsterNet decoding and CLI regression checks; no downloads needed."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

import asternet_ocr
import main


class AsterNetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.image = self.folder / "sample.png"
        Image.new("RGB", (240, 40), "white").save(self.image)

    def test_ctc_collapses_repeats_but_blank_allows_double_character(self):
        tokens = [0, 1, 1, 0, 1, 2, 2, 0, 3]
        values = np.full((len(tokens), 1, 4), -100.0, dtype=np.float32)
        for step, token in enumerate(tokens):
            values[step, 0, token] = 0
        result = asternet_ocr.decode_ctc(values, ["<blank>", "a", "b", "\u1ebf"])
        self.assertEqual(result["text"], "aab\u1ebf")
        self.assertAlmostEqual(result["confidence"], 1)

    def test_blank_prediction_is_empty(self):
        result = asternet_ocr.decode_ctc(np.array([[[0., -10.]]]), ["<blank>", "a"])
        self.assertEqual(result, {"text": "", "confidence": 0.0})

    def test_mismatched_output_and_nonfinite_values_fail(self):
        for values in (np.zeros((1, 3, 2)), np.zeros((1, 1, 3)), np.full((1, 1, 2), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(RuntimeError):
                asternet_ocr.decode_ctc(values, ["<blank>", "a"])

    def test_preprocessing_shape_range_and_dark_background(self):
        dark = Image.new("RGB", (240, 40), "black")
        ImageDraw.Draw(dark).rectangle((20, 10, 60, 30), fill="white")
        tensor = asternet_ocr.preprocess(dark)
        self.assertEqual(tensor.shape[:3], (1, 3, 32))
        self.assertEqual(tensor.shape[3] % 4, 0)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreaterEqual(float(tensor.min()), 0)
        self.assertLessEqual(float(tensor.max()), 1)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1)

    def test_bad_inputs_fail_before_model_download(self):
        with patch("asternet_ocr._model_files", side_effect=AssertionError("must not download")):
            with self.assertRaises(FileNotFoundError):
                asternet_ocr.extract_text(self.folder / "missing.png")
            with self.assertRaises(ValueError):
                asternet_ocr.extract_text(self.image, mode="unknown")
            invalid = self.folder / "invalid.png"
            invalid.write_text("not an image")
            with self.assertRaises(OSError):
                asternet_ocr.extract_text(invalid)

    def test_explicit_local_folder_requires_all_model_files(self):
        with self.assertRaisesRegex(FileNotFoundError, "asternet.onnx"):
            asternet_ocr._model_files(self.folder)

    def test_document_detection_uses_bgr_and_returns_ordered_crops(self):
        left = [[10, 10], [80, 10], [80, 30], [10, 30]]
        right = [[100, 12], [180, 12], [180, 32], [100, 32]]
        bottom = [[10, 60], [180, 60], [180, 80], [10, 80]]
        detector = Mock(return_value=(np.float32([bottom, right, left]), 0.01))
        factory = Mock(return_value=detector)
        with patch.dict("sys.modules", {"rapidocr_onnxruntime": SimpleNamespace(RapidOCR=factory)}):
            crops = asternet_ocr._document_crops(Image.new("RGB", (200, 100), (250, 70, 20)))
        self.assertEqual([polygon for _, polygon in crops], [left, right, bottom])
        self.assertEqual(detector.call_args.args[0][0, 0].tolist(), [20, 70, 250])
        self.assertFalse(detector.call_args.kwargs["use_rec"])
        self.assertEqual(crops[0][0].getpixel((0, 0)), (250, 70, 20))

    def test_document_without_detected_text_returns_no_crops(self):
        detector = Mock(return_value=(None, None))
        factory = Mock(return_value=detector)
        with patch.dict("sys.modules", {"rapidocr_onnxruntime": SimpleNamespace(RapidOCR=factory)}):
            self.assertEqual(asternet_ocr._document_crops(Image.new("RGB", (100, 100))), [])

    def test_main_calls_module_and_saves_utf8_outputs(self):
        result = {"engine": "asternet", "text": "Ti\u1ebfng Vi\u1ec7t", "lines": []}
        argv = ["main.py", str(self.image), "--engine", "asternet", "--asternet-mode", "line",
                "--asternet-model-dir", str(self.folder), "--output", str(self.folder / "out")]
        with patch("sys.argv", argv), patch("asternet_ocr.extract_text", return_value=result) as extract, \
                patch("main.extract_text", side_effect=AssertionError("PaddleOCR must not run")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main.main(), 0)
        extract.assert_called_once_with(self.image.resolve(), mode="line", model_dir=self.folder)
        output = self.folder / "out" / "asternet"
        self.assertEqual((output / "sample.txt").read_text(encoding="utf-8"), result["text"])
        self.assertEqual(json.loads((output / "sample.json").read_text(encoding="utf-8")), result)

    def test_missing_dependency_reports_asternet_install_command(self):
        argv = ["main.py", str(self.image), "--engine", "asternet"]
        with patch("sys.argv", argv), patch("asternet_ocr.extract_text", side_effect=ImportError("rapidocr_onnxruntime")), \
                redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main.main(), 1)
        self.assertIn("python -m pip install -r requirements-asternet.txt", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

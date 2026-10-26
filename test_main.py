"""Regression checks for structured extraction; no OCR model downloads needed."""

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

from main import detect_qr_code, parse_aadhaar, recognize_id_details


def raw(*texts):
    return {"lines": [{"text": text, "confidence": 0.99} for text in texts]}


class AadhaarParsingTests(unittest.TestCase):
    def test_fields_are_extracted_from_input(self):
        source = raw(
            "AADHAAR", "Kavita Sharma", "DOB: 07/11/1990", "FEMALE",
            "2345 6789 0123", "VID: 1234 5678 9012 3456",
            "Issue Date: 03/04/2021", "Download Date: 05/06/2022",
            "Address:", "C/O Rajesh Sharma, 45, Shastri Nagar,",
            "Jaipur, Jaipur,", "Rajasthan - 302001",
        )
        source["lines"].append({
            "text": "कविता शर्मा", "confidence": 0.98, "source": "hindi_name",
        })
        result = parse_aadhaar(source, qr_present=True)
        self.assertEqual(result["name"], "Kavita Sharma")
        self.assertEqual(result["name_hindi"], "कविता शर्मा")
        self.assertEqual(result["date_of_birth"], "1990-11-07")
        self.assertEqual(result["gender"], "Female")
        self.assertEqual(result["aadhaar_number"], "234567890123")
        self.assertEqual(result["virtual_id"], "1234567890123456")
        self.assertEqual(result["issue_date"], "2021-04-03")
        self.assertEqual(result["download_date"], "2022-06-05")
        self.assertEqual(result["address"], {
            "care_of": "Rajesh Sharma", "house_number": "45",
            "locality": "Shastri Nagar", "city": "Jaipur", "district": "Jaipur",
            "state": "Rajasthan", "pin_code": "302001", "country": "India",
        })
        self.assertEqual(result["document_type"], "Aadhaar")
        self.assertEqual(result["issuing_authority"], "UIDAI")
        self.assertTrue(result["qr_code_present"])

    def test_no_values_are_filled_from_the_example(self):
        result = parse_aadhaar(raw("AADHAAR"))
        for key, value in result.items():
            if key not in ("document_type", "issuing_authority", "address", "qr_code_present"):
                self.assertEqual(value, "", key)
        self.assertTrue(all(value == "" for value in result["address"].values()))
        self.assertFalse(result["qr_code_present"])

    def test_invalid_or_unlabelled_dates_are_not_used(self):
        result = parse_aadhaar(raw(
            "AADHAAR", "Name: Ravi Singh", "DOB: 31/02/2000",
            "Issue Date: 99/01/2020", "12/04/2022", "Year of Birth: 2000",
        ))
        self.assertEqual(result["name"], "Ravi Singh")
        for key in ("date_of_birth", "issue_date", "download_date"):
            self.assertEqual(result[key], "")

    def test_virtual_id_does_not_become_aadhaar_number(self):
        result = parse_aadhaar(raw("AADHAAR", "VID: 1234 5678 9012 3456"))
        self.assertEqual(result["virtual_id"], "1234567890123456")
        self.assertEqual(result["aadhaar_number"], "")

    def test_low_confidence_fields_stay_empty(self):
        source = raw("AADHAAR")
        source["lines"].extend([
            {"text": "DOB: 01/02/2000", "confidence": 0.4},
            {"text": "गलत नाम", "source": "hindi_name", "confidence": 0.3},
        ])
        result = parse_aadhaar(source)
        self.assertEqual(result["date_of_birth"], "")
        self.assertEqual(result["name_hindi"], "")

    def test_unknown_document_is_not_assumed_to_be_aadhaar(self):
        result = parse_aadhaar(raw("Passport", "Name: Ravi Singh"))
        self.assertEqual(result["document_type"], "")
        self.assertEqual(result["issuing_authority"], "")

    def test_damaged_heading_does_not_discard_other_card_fields(self):
        result = parse_aadhaar(raw(
            "unreadable heading", "Address:", "C/O Rajesh Sharma, Jaipur, Rajasthan - 302001",
            "2345 6789 0123", "Kavita Sharma", "/D07/11/1990", "/FEMALE",
            "2345 6789 0123", "Download Date: 05/12/2019",
        ))
        self.assertEqual(result["document_type"], "Aadhaar")
        self.assertEqual(result["name"], "Kavita Sharma")
        self.assertEqual(result["date_of_birth"], "1990-11-07")
        self.assertEqual(result["gender"], "Female")
        self.assertEqual(result["aadhaar_number"], "234567890123")
        self.assertEqual(result["address"]["state"], "Rajasthan")
        self.assertEqual(result["download_date"], "2019-12-05")

    def test_uidai_domain_with_missing_letter_is_recognized(self):
        result = parse_aadhaar(raw("idai gov.in", "2345 6789 0123"))
        self.assertEqual(result["document_type"], "Aadhaar")
        self.assertEqual(result["aadhaar_number"], "234567890123")

    def test_a_twelve_digit_number_alone_is_insufficient_document_evidence(self):
        result = parse_aadhaar(raw("Receipt", "2345 6789 0123"))
        self.assertEqual(result["document_type"], "")
        self.assertEqual(result["issuing_authority"], "")

    def test_unlabelled_date_does_not_become_dob(self):
        result = parse_aadhaar(raw("UIDAI", "Kavita Sharma", "07/11/1990", "FEMALE"))
        self.assertEqual(result["date_of_birth"], "")

    def test_short_address_does_not_invent_district(self):
        result = parse_aadhaar(raw(
            "UIDAI", "Address:", "Shastri Nagar, Jaipur, Rajasthan - 302001",
        ))
        self.assertEqual(result["address"]["city"], "Jaipur")
        self.assertEqual(result["address"]["district"], "")

    def test_wrapped_address_and_post_office_suffix(self):
        result = parse_aadhaar(raw(
            "UIDAI", "Address:", "D/O: Rajesh Sharma, gopal ganj / upper",
            "bazar, near shani mandir, upper bazar ranchi,",
            "Ranchi G.P.O., Ranchi, Jharkhand, 834001",
        ))
        self.assertEqual(result["address"]["care_of"], "Rajesh Sharma")
        self.assertEqual(result["address"]["house_number"], "")
        self.assertEqual(result["address"]["locality"],
                         "Gopal Ganj / Upper Bazar, Near Shani Mandir, Upper Bazar Ranchi")
        self.assertEqual(result["address"]["city"], "Ranchi")
        self.assertEqual(result["address"]["district"], "Ranchi")

    def test_enrolment_letter_with_labelled_address_without_address_heading(self):
        result = parse_aadhaar(raw(
            "Unique Identification Authority of India", "Name: Demo Person",
            "DOB: 01/01/2000", "MALE", "C/O: Demo Guardian,",
            "VTC: Flat No 42", "PO: Jaipur", "Sub District: Jaipur",
            "District: Jaipur", "State: Rajasthan", "PIN Code: 302001,",
            "Mobile: 9876543210", "2345 6789 0123",
        ))
        self.assertEqual(result["address"], {
            "care_of": "Demo Guardian", "house_number": "42", "locality": "",
            "city": "Jaipur", "district": "Jaipur", "state": "Rajasthan",
            "pin_code": "302001", "country": "India",
        })

    def test_sample_number_is_extracted_as_printed(self):
        result = parse_aadhaar(raw("UIDAI", "1234 5678 9101"))
        self.assertEqual(result["aadhaar_number"], "123456789101")

    def test_blank_image_has_no_qr(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "blank.png"
            cv2.imwrite(str(path), np.full((500, 500, 3), 255, dtype=np.uint8))
            self.assertFalse(detect_qr_code(path))

    def test_hindi_name_crop_can_overlap_english_name_box(self):
        import numpy as np
        image = np.repeat(np.arange(220, dtype=np.uint8)[:, None, None], 400, axis=1)
        image = np.repeat(image, 3, axis=2)
        source = raw("Government of India", "unreadable", "Ravi Singh", "DOB: 01/02/2000")
        for line, box in zip(source["lines"], [
            [[100, 30], [300, 55]], [[100, 110], [210, 142]],
            [[100, 139], [220, 168]], [[100, 170], [300, 195]],
        ]):
            line["polygon"] = box
        inputs = []

        class Recognizer:
            def __init__(self, **kwargs):
                pass

            def predict(self, image):
                inputs.append(image)
                return [{"rec_text": "रवि सिंह", "rec_score": .99}]

            def close(self):
                pass

        with patch.dict("sys.modules", {"paddleocr": SimpleNamespace(TextRecognition=Recognizer)}):
            extra = recognize_id_details(image, source["lines"])
        self.assertEqual(extra[0]["source"], "hindi_name")
        # The selected pixels belong to the Hindi row, not the government heading.
        self.assertGreater(float(inputs[0].mean()), 100)
        self.assertLess(float(inputs[0].mean()), 145)

    def test_fragmented_vertical_label_and_date_are_read_together(self):
        import numpy as np
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        source = raw("date digits", "download label")
        source["lines"][0]["polygon"] = [[10, 50], [25, 120]]
        source["lines"][1]["polygon"] = [[10, 180], [25, 250]]
        inputs = []

        class Recognizer:
            def __init__(self, **kwargs):
                pass

            def predict(self, images, **kwargs):
                inputs.extend(images)
                return [{"rec_text": "Download Date: 05/06/2022", "rec_score": .99}
                        for _ in images]

            def close(self):
                pass

        with patch.dict("sys.modules", {"paddleocr": SimpleNamespace(TextRecognition=Recognizer)}):
            extra = recognize_id_details(image, source["lines"])
        self.assertEqual(len(inputs), 2)  # Both directions of one combined strip.
        self.assertGreater(inputs[0].shape[1], 190)
        self.assertEqual(parse_aadhaar(raw("AADHAAR", extra[0]["text"]))["download_date"], "2022-06-05")


if __name__ == "__main__":
    unittest.main()

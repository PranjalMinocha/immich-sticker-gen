import json
import os
import tempfile
import unittest
from typing import Optional

from PIL import Image

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion_checks import validate_sample
from ingestion_config import load_config


class IngestionChecksTests(unittest.TestCase):
    def _write_image(self, path: str, size: tuple[int, int] = (128, 128), color: tuple[int, int, int] = (80, 120, 160)):
        image = Image.new("RGB", size=size, color=color)
        image.save(path, format="JPEG")

    def _write_annotation(self, path: str, bbox: Optional[list[float]] = None):
        bbox = bbox or [10, 10, 40, 40]
        payload = {"annotations": [{"bbox": bbox}]}
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle)

    def test_valid_pair_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "sa_1.jpg")
            annotation_path = os.path.join(temp_dir, "sa_1.json")
            self._write_image(image_path)
            self._write_annotation(annotation_path)

            result = validate_sample(image_path, annotation_path, load_config(), set(), [])
            self.assertEqual(result.hard_fail_reasons, [])

    def test_missing_annotation_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "sa_2.jpg")
            annotation_path = os.path.join(temp_dir, "sa_2.json")
            self._write_image(image_path)

            result = validate_sample(image_path, annotation_path, load_config(), set(), [])
            self.assertIn("missing_or_empty_annotation", result.hard_fail_reasons)

    def test_invalid_json_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "sa_3.jpg")
            annotation_path = os.path.join(temp_dir, "sa_3.json")
            self._write_image(image_path)
            with open(annotation_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("{broken json")

            result = validate_sample(image_path, annotation_path, load_config(), set(), [])
            self.assertIn("annotation_json_invalid", result.hard_fail_reasons)

    def test_exact_duplicate_is_hard_fail_on_second(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path_a = os.path.join(temp_dir, "sa_4.jpg")
            image_path_b = os.path.join(temp_dir, "sa_5.jpg")
            annotation_path_a = os.path.join(temp_dir, "sa_4.json")
            annotation_path_b = os.path.join(temp_dir, "sa_5.json")
            self._write_image(image_path_a)
            self._write_image(image_path_b)
            self._write_annotation(annotation_path_a)
            self._write_annotation(annotation_path_b)

            seen_sha256 = set()
            seen_ahash = []
            first = validate_sample(image_path_a, annotation_path_a, load_config(), seen_sha256, seen_ahash)
            second = validate_sample(image_path_b, annotation_path_b, load_config(), seen_sha256, seen_ahash)

            self.assertEqual(first.hard_fail_reasons, [])
            self.assertIn("exact_duplicate_image", second.hard_fail_reasons)


if __name__ == "__main__":
    unittest.main()

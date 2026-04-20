import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from build_live_drift_detector import annotation_to_vector, parse_s3_uri


class BuildLiveDriftDetectorTests(unittest.TestCase):
    def test_parse_s3_uri(self):
        bucket, key = parse_s3_uri("s3://bucket-a/path/to/file.json")
        self.assertEqual(bucket, "bucket-a")
        self.assertEqual(key, "path/to/file.json")

    def test_parse_s3_uri_rejects_invalid(self):
        with self.assertRaises(ValueError):
            parse_s3_uri("http://bucket/key")
        with self.assertRaises(ValueError):
            parse_s3_uri("s3://missing-key")

    def test_annotation_to_vector_with_center_fallback(self):
        ann = {"bbox": [10, 20, 30, 40]}
        vector = annotation_to_vector(ann)
        self.assertIsNotNone(vector)
        self.assertEqual(vector.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()

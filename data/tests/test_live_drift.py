import os
import shutil
import tempfile
import unittest

import numpy as np

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live_drift import download_detector_artifact, extract_request_features, upload_detector_artifact


class FakeS3Client:
    def __init__(self, root: str):
        self.root = root

    def _path(self, bucket: str, key: str) -> str:
        return os.path.join(self.root, bucket, key)

    def upload_file(self, local_path: str, bucket: str, key: str) -> None:
        target = self._path(bucket, key)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(local_path, target)

    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        source = self._path(bucket, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(source, local_path)


class LiveDriftTests(unittest.TestCase):
    def test_extract_request_features_from_json_strings(self):
        vector = extract_request_features("[10, 20, 30, 40]", "[[15, 25], [18, 30]]")
        self.assertIsNotNone(vector)
        self.assertEqual(vector.shape, (1, 9))
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(float(vector[0][8]), 2.0)

    def test_extract_request_features_rejects_invalid_bbox(self):
        self.assertIsNone(extract_request_features("[1,2,3]", "[[1,2]]"))
        self.assertIsNone(extract_request_features("[1,2,-3,4]", "[[1,2]]"))
        self.assertIsNone(extract_request_features("not-json", "[[1,2]]"))

    def test_detector_artifact_upload_and_download_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_s3 = FakeS3Client(root=temp_dir)
            detector_src = os.path.join(temp_dir, "detector_src", "cd")
            os.makedirs(detector_src, exist_ok=True)
            marker_path = os.path.join(detector_src, "marker.txt")
            with open(marker_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("ok")

            upload_detector_artifact(fake_s3, "bucket-a", "drift/cd.tar.gz", detector_src)

            cache_root = os.path.join(temp_dir, "cache")
            detector_downloaded = download_detector_artifact(fake_s3, "bucket-a", "drift/cd.tar.gz", cache_root)

            self.assertTrue(os.path.isdir(detector_downloaded))
            self.assertTrue(os.path.exists(os.path.join(detector_downloaded, "marker.txt")))


if __name__ == "__main__":
    unittest.main()

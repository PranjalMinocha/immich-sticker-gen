import json
import os
import shutil
import tempfile
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import compile_retraining_dataset as module


class FakeS3Client:
    def __init__(self, root: str):
        self.root = root

    def _path(self, bucket: str, key: str) -> str:
        return os.path.join(self.root, bucket, key)

    def head_object(self, Bucket: str, Key: str):
        path = self._path(Bucket, Key)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    def copy_object(self, Bucket: str, CopySource, Key: str):
        src = self._path(CopySource["Bucket"], CopySource["Key"])
        dst = self._path(Bucket, Key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str = "application/octet-stream"):
        dst = self._path(Bucket, Key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as handle:
            handle.write(Body)


class CompileRetrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="retrain_compile_")
        self.bucket = "bucket-a"
        os.makedirs(os.path.join(self.temp_dir.name, self.bucket, "user_uploads"), exist_ok=True)
        self.fake_s3 = FakeS3Client(self.temp_dir.name)

        module.RAW_BUCKET = self.bucket
        module._s3_client = lambda: self.fake_s3

        for idx in range(3):
            img_path = os.path.join(self.temp_dir.name, self.bucket, "user_uploads", f"img_{idx}.jpg")
            with open(img_path, "wb") as handle:
                handle.write(b"fakejpeg")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compile_retraining_dataset_writes_manifests_and_metadata(self):
        rows = []
        for idx in range(3):
            rows.append(
                {
                    "generationId": f"00000000-0000-0000-0000-00000000000{idx}",
                    "bbox": "[10, 20, 30, 40]",
                    "pointCoords": "[[20, 30]]",
                    "userSavedMask": '{"size":[4,4],"counts":[0,16]}',
                    "originalPath": f"user_uploads/img_{idx}.jpg",
                }
            )

        result = module.compile_retraining_dataset(rows, "run123")

        self.assertEqual(result.total_rows, 3)
        self.assertEqual(result.train_count, 2)
        self.assertEqual(result.val_count, 1)
        self.assertEqual(result.skipped_count, 0)
        self.assertTrue(result.train_manifest_s3_uri.endswith("/retraining_runs/run123/dataset_manifests/train_manifest.csv"))

        metadata_path = os.path.join(self.temp_dir.name, self.bucket, "retraining_runs", "run123", "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["acceptedRows"], 3)
        self.assertEqual(metadata["trainCount"], 2)
        self.assertEqual(metadata["valCount"], 1)

    def test_compile_requires_enough_accepted_rows(self):
        rows = [
            {
                "generationId": "00000000-0000-0000-0000-000000000001",
                "bbox": "[10, 20, 30, 40]",
                "pointCoords": "[[20, 30]]",
                "userSavedMask": '{"size":[4,4],"counts":[0,16]}',
                "originalPath": "user_uploads/img_0.jpg",
            }
        ]

        with self.assertRaises(ValueError):
            module.compile_retraining_dataset(rows, "run_single")


if __name__ == "__main__":
    unittest.main()

import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion_config import load_config, validate_storage_config


class IngestionConfigTests(unittest.TestCase):
    def test_expected_backend_mismatch_raises(self):
        prev_storage = os.environ.get("STORAGE_BACKEND")
        prev_expected = os.environ.get("INGEST_EXPECTED_BACKEND")
        try:
            os.environ["STORAGE_BACKEND"] = "local"
            os.environ["INGEST_EXPECTED_BACKEND"] = "s3"
            cfg = load_config()
            with self.assertRaises(ValueError):
                validate_storage_config(cfg)
        finally:
            if prev_storage is None:
                os.environ.pop("STORAGE_BACKEND", None)
            else:
                os.environ["STORAGE_BACKEND"] = prev_storage
            if prev_expected is None:
                os.environ.pop("INGEST_EXPECTED_BACKEND", None)
            else:
                os.environ["INGEST_EXPECTED_BACKEND"] = prev_expected

    def test_strict_storage_backend_requires_s3(self):
        prev_storage = os.environ.get("STORAGE_BACKEND")
        prev_strict = os.environ.get("STRICT_STORAGE_BACKEND")
        try:
            os.environ["STORAGE_BACKEND"] = "local"
            os.environ["STRICT_STORAGE_BACKEND"] = "true"
            cfg = load_config()
            with self.assertRaises(ValueError):
                validate_storage_config(cfg)
        finally:
            if prev_storage is None:
                os.environ.pop("STORAGE_BACKEND", None)
            else:
                os.environ["STORAGE_BACKEND"] = prev_storage
            if prev_strict is None:
                os.environ.pop("STRICT_STORAGE_BACKEND", None)
            else:
                os.environ["STRICT_STORAGE_BACKEND"] = prev_strict


if __name__ == "__main__":
    unittest.main()

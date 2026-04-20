import io
import json
import os
import tarfile
import tempfile
import unittest

from PIL import Image

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest_data import ingest_archive


def _jpg_bytes(size=(128, 128), color=(90, 120, 150)) -> bytes:
    image = Image.new("RGB", size=size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class IngestLocalModeTests(unittest.TestCase):
    def test_ingest_archive_routes_files_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, "sample.tar")
            local_store = os.path.join(temp_dir, "store")

            previous_backend = os.environ.get("STORAGE_BACKEND")
            previous_store = os.environ.get("LOCAL_STORE_ROOT")
            previous_hard_fail = os.environ.get("HARD_FAIL_RATE_MAX")
            previous_strict = os.environ.get("STRICT_STORAGE_BACKEND")
            previous_expected = os.environ.get("INGEST_EXPECTED_BACKEND")

            os.environ["STORAGE_BACKEND"] = "local"
            os.environ["LOCAL_STORE_ROOT"] = local_store
            os.environ["HARD_FAIL_RATE_MAX"] = "1.0"
            os.environ["STRICT_STORAGE_BACKEND"] = "false"
            os.environ.pop("INGEST_EXPECTED_BACKEND", None)

            try:
                with tarfile.open(archive_path, "w") as archive:
                    good_jpg = _jpg_bytes()
                    good_json = json.dumps({"annotations": [{"bbox": [5, 5, 50, 50]}]}).encode("utf-8")
                    missing_pair_jpg = _jpg_bytes(color=(30, 30, 30))

                    info_img = tarfile.TarInfo(name="sa_good.jpg")
                    info_img.size = len(good_jpg)
                    archive.addfile(info_img, io.BytesIO(good_jpg))

                    info_ann = tarfile.TarInfo(name="sa_good.json")
                    info_ann.size = len(good_json)
                    archive.addfile(info_ann, io.BytesIO(good_json))

                    info_missing = tarfile.TarInfo(name="sa_missing.jpg")
                    info_missing.size = len(missing_pair_jpg)
                    archive.addfile(info_missing, io.BytesIO(missing_pair_jpg))

                summary = ingest_archive(archive_path)

                self.assertEqual(summary["total_samples"], 2)
                self.assertTrue(os.path.exists(os.path.join(local_store, "images", "sa_good.jpg")))
                self.assertTrue(os.path.exists(os.path.join(local_store, "annotations", "sa_good.json")))
                self.assertTrue(
                    os.path.exists(
                        os.path.join(
                            local_store,
                            "quarantine",
                            "missing_annotation_for_image",
                            "sa_missing.jpg",
                        )
                    )
                )

                manifests_dir = os.path.join(local_store, "quality", "manifests")
                reports_dir = os.path.join(local_store, "quality", "reports")
                self.assertTrue(os.listdir(manifests_dir))
                self.assertTrue(os.listdir(reports_dir))
            finally:
                if previous_backend is None:
                    os.environ.pop("STORAGE_BACKEND", None)
                else:
                    os.environ["STORAGE_BACKEND"] = previous_backend

                if previous_store is None:
                    os.environ.pop("LOCAL_STORE_ROOT", None)
                else:
                    os.environ["LOCAL_STORE_ROOT"] = previous_store

                if previous_hard_fail is None:
                    os.environ.pop("HARD_FAIL_RATE_MAX", None)
                else:
                    os.environ["HARD_FAIL_RATE_MAX"] = previous_hard_fail

                if previous_strict is None:
                    os.environ.pop("STRICT_STORAGE_BACKEND", None)
                else:
                    os.environ["STRICT_STORAGE_BACKEND"] = previous_strict

                if previous_expected is None:
                    os.environ.pop("INGEST_EXPECTED_BACKEND", None)
                else:
                    os.environ["INGEST_EXPECTED_BACKEND"] = previous_expected


if __name__ == "__main__":
    unittest.main()

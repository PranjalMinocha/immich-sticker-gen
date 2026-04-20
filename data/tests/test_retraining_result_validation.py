import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retraining_result_validation import validate_training_result


class RetrainingResultValidationTests(unittest.TestCase):
    def test_valid_result_passes(self):
        payload = {
            "status": "passed",
            "mlflow": {
                "runId": "abc123",
                "registered": True,
                "modelName": "immich-sticker-mobilesam",
                "modelVersion": "12",
            },
        }
        ok, reason = validate_training_result(payload)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_quality_gate_failure_is_rejected(self):
        payload = {"status": "failed", "mlflow": {"runId": "abc"}}
        ok, reason = validate_training_result(payload)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("quality_gate_failed"))

    def test_missing_registry_info_is_rejected(self):
        payload = {
            "status": "passed",
            "mlflow": {
                "runId": "abc123",
                "registered": False,
            },
        }
        ok, reason = validate_training_result(payload)
        self.assertFalse(ok)
        self.assertEqual(reason, "model_not_registered")


if __name__ == "__main__":
    unittest.main()

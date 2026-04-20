import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from retraining_result_contract import validate_result_payload


class RetrainingResultContractTests(unittest.TestCase):
    def _valid_payload(self):
        return {
            "status": "passed",
            "metrics": {"dice": 0.8, "iou": 0.7, "runtimeSeconds": 123},
            "qualityGate": {"passed": True},
            "mlflow": {
                "trackingUri": "http://mlflow:5000",
                "runId": "abc123",
                "registered": True,
                "modelName": "immich-sticker-mobilesam",
                "modelVersion": "7",
            },
        }

    def test_valid_payload_has_no_errors(self):
        errors = validate_result_payload(self._valid_payload())
        self.assertEqual(errors, [])

    def test_requires_model_fields_when_registered(self):
        payload = self._valid_payload()
        payload["mlflow"].pop("modelVersion")
        errors = validate_result_payload(payload)
        self.assertTrue(any("modelVersion" in error for error in errors))

    def test_require_failed_checks_status(self):
        errors = validate_result_payload(self._valid_payload(), require_passed=False)
        self.assertTrue(any("expected failed" in error for error in errors))

    def test_require_not_registered(self):
        errors = validate_result_payload(self._valid_payload(), require_registered=False)
        self.assertTrue(any("expected mlflow.registered=false" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retraining_result_validation import validate_training_result


class RetrainingResultValidationTests(unittest.TestCase):
    # Helpers matching the real payload structure from train.py:
    #   - "metrics" is at top level (dice, iou, runtimeSeconds, …)
    #   - "qualityGate" contains pass/fail flags (passed, passDice, passIou, …)
    def _valid_payload(self):
        return {
            "status": "passed",
            "metrics": {"dice": 0.85, "iou": 0.82, "runtimeSeconds": 120},
            "mlflow": {
                "runId": "abc123",
                "registered": True,
                "modelName": "immich-sticker-mobilesam",
                "modelVersion": "12",
            },
            "qualityGate": {
                "passed": True,
                "passDice": True,
                "passIou": True,
                "passRuntime": True,
            },
        }

    def test_valid_result_passes(self):
        ok, reason = validate_training_result(self._valid_payload())
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_quality_gate_block_missing_is_rejected(self):
        payload = self._valid_payload()
        del payload["qualityGate"]
        ok, reason = validate_training_result(payload)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_quality_gate_block")

    def test_quality_gate_not_passed_is_rejected(self):
        payload = self._valid_payload()
        payload["qualityGate"]["passed"] = False
        payload["qualityGate"]["reason"] = "iou_too_low"
        ok, reason = validate_training_result(payload)
        self.assertFalse(ok)
        self.assertIn("iou_too_low", reason)

    def test_missing_metrics_is_rejected(self):
        payload = self._valid_payload()
        del payload["metrics"]
        ok, reason = validate_training_result(payload)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_metrics")

    def test_empty_metrics_is_rejected(self):
        payload = self._valid_payload()
        payload["metrics"] = {}
        ok, reason = validate_training_result(payload)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_metrics")

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

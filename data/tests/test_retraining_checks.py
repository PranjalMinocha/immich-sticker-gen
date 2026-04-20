import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retraining_checks import load_quality_config, should_block_batch, summarize_validation, validate_rows


class RetrainingChecksTests(unittest.TestCase):
    def _valid_row(self, generation_id: int = 1):
        return {
            "generation_id": generation_id,
            "user_id": 101,
            "image_id": 202,
            "bbox": "[10, 20, 30, 40]",
            "point_coords": "[[20, 30]]",
            "user_saved_mask": "RLE_ABC",
            "edited_pixels": 100,
            "num_tries": 2,
            "processing_time_ms": 1200,
            "generated_at": "2026-04-19T12:00:00Z",
        }

    def test_valid_row_passes(self):
        cfg = load_quality_config()
        validated = validate_rows([self._valid_row()], cfg)
        self.assertEqual(validated[0]["status"], "pass")
        self.assertEqual(validated[0]["hard_fail_reasons"], [])

    def test_invalid_bbox_is_hard_fail(self):
        cfg = load_quality_config()
        row = self._valid_row()
        row["bbox"] = "[10, 20, -1, 40]"
        validated = validate_rows([row], cfg)
        self.assertEqual(validated[0]["status"], "hard_fail")
        self.assertTrue(any(reason.startswith("invalid_bbox") for reason in validated[0]["hard_fail_reasons"]))

    def test_duplicate_generation_id_is_hard_fail_on_second(self):
        cfg = load_quality_config()
        row_a = self._valid_row(generation_id=100)
        row_b = self._valid_row(generation_id=100)
        row_b["image_id"] = 999

        validated = validate_rows([row_a, row_b], cfg)
        self.assertEqual(validated[0]["status"], "pass")
        self.assertEqual(validated[1]["status"], "hard_fail")
        self.assertIn("duplicate_generation_id", validated[1]["hard_fail_reasons"])

    def test_summary_and_blocking(self):
        cfg = load_quality_config()
        good = self._valid_row(generation_id=1)
        bad = self._valid_row(generation_id=2)
        bad["user_saved_mask"] = ""

        validated = validate_rows([good, bad], cfg)
        summary = summarize_validation(validated)
        self.assertEqual(summary["total_candidates"], 2)
        self.assertEqual(summary["hard_fail_count"], 1)

        strict_cfg = cfg.__class__(
            max_edited_pixels_warn=cfg.max_edited_pixels_warn,
            max_num_tries_warn=cfg.max_num_tries_warn,
            max_processing_time_ms_warn=cfg.max_processing_time_ms_warn,
            max_hard_fail_rate=0.1,
            max_soft_warn_rate=cfg.max_soft_warn_rate,
            min_accepted_batch_size=2,
        )
        blocked, reasons = should_block_batch(summary, strict_cfg)
        self.assertTrue(blocked)
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()

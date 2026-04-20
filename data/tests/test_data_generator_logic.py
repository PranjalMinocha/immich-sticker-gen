import json
import os
import unittest

import numpy as np

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data_generator as module


class DataGeneratorLogicTests(unittest.TestCase):
    def test_expand_only_bbox_never_shrinks(self):
        rng_state = module.random.getstate()
        try:
            module.random.seed(123)
            original = [100.0, 80.0, 60.0, 40.0]
            expanded = module.apply_bbox_noise_expand_only(original, image_width=1000, image_height=1000)
            self.assertLessEqual(expanded[0], original[0])
            self.assertLessEqual(expanded[1], original[1])
            self.assertGreaterEqual(expanded[2], original[2])
            self.assertGreaterEqual(expanded[3], original[3])
        finally:
            module.random.setstate(rng_state)

    def test_expand_only_bbox_stays_inside_image(self):
        rng_state = module.random.getstate()
        try:
            module.random.seed(99)
            original = [2.0, 3.0, 5.0, 4.0]
            expanded = module.apply_bbox_noise_expand_only(original, image_width=10, image_height=10)
            x, y, w, h = expanded
            self.assertGreaterEqual(x, 0.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(x + w, 10.0)
            self.assertLessEqual(y + h, 10.0)
        finally:
            module.random.setstate(rng_state)

    def test_rle_decode_encode_round_trip(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 2:5] = True
        rle = module._encode_uncompressed_rle(mask)
        decoded = module._decode_rle_mask(rle)
        self.assertTrue(np.array_equal(mask, decoded))

    def test_partition_total_preserves_sum(self):
        np_state = np.random.get_state()
        try:
            np.random.seed(7)
            parts = module._partition_total(1234, 4)
            self.assertEqual(len(parts), 4)
            self.assertEqual(sum(parts), 1234)
            self.assertTrue(all(p >= 0 for p in parts))
        finally:
            np.random.set_state(np_state)

    def test_annotation_polygon_decode(self):
        annotation = {
            "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
            "bbox": [1, 1, 3, 3],
        }
        mask = module._decode_annotation_mask(annotation, image_height=6, image_width=6)
        self.assertEqual(mask.shape, (6, 6))
        self.assertGreater(mask.sum(), 0)

    def test_diff_pixels_logic(self):
        gt = np.zeros((3, 3), dtype=bool)
        gt[0, 0] = True
        gt[1, 1] = True

        model = np.zeros((3, 3), dtype=bool)
        model[0, 0] = True
        model[2, 2] = True

        diff = int(np.count_nonzero(np.logical_xor(gt, model)))
        self.assertEqual(diff, 2)


if __name__ == "__main__":
    unittest.main()

import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retraining_trigger_logic import should_trigger_retraining


class RetrainingTriggerLogicTests(unittest.TestCase):
    def test_below_threshold_does_not_trigger(self):
        self.assertFalse(should_trigger_retraining(4999, 5000))

    def test_equal_or_above_threshold_triggers(self):
        self.assertTrue(should_trigger_retraining(5000, 5000))
        self.assertTrue(should_trigger_retraining(7000, 5000))


if __name__ == "__main__":
    unittest.main()

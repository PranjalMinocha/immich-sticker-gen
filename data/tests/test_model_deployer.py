import os
import unittest
from unittest.mock import patch

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_deployer import ping_serving_reload


class _Response:
    def __init__(self, code, text):
        self.status_code = code
        self.text = text


class ModelDeployerTests(unittest.TestCase):
    def test_ping_reload_unset_url(self):
        ok, detail = ping_serving_reload("")
        self.assertFalse(ok)
        self.assertEqual(detail, "reload_url_unset")

    @patch("model_deployer.requests.post", return_value=_Response(200, "ok"))
    def test_ping_reload_success(self, _mock_post):
        ok, detail = ping_serving_reload("http://serving/reload", "abc")
        self.assertTrue(ok)
        self.assertEqual(detail, "ok")

    @patch("model_deployer.requests.post", return_value=_Response(500, "boom"))
    def test_ping_reload_failure(self, _mock_post):
        ok, detail = ping_serving_reload("http://serving/reload", "abc")
        self.assertFalse(ok)
        self.assertTrue(detail.startswith("http_500"))


if __name__ == "__main__":
    unittest.main()

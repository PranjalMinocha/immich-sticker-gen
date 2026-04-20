import os
import unittest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import seed_synthetic_users as module


class FakeCursor:
    def __init__(self):
        self.users = []
        self._last_row = {"count": 0}
        self.rowcount = 0

    def execute(self, query, params=None):
        query_text = " ".join(query.split()).lower()
        self.rowcount = 0

        if query_text.startswith('select count(*) as count from "user" where "email" like %s and "mltrainingoptin" = %s;'):
            prefix_like, ml_opt_in = params
            prefix = prefix_like[:-1]
            count = sum(1 for user in self.users if user["email"].startswith(prefix) and user["mlTrainingOptIn"] == ml_opt_in)
            self._last_row = {"count": count}
            return

        if query_text.startswith('insert into "user" ("name", "email", "mltrainingoptin") values'):
            name, email, ml_opt_in = params
            if any(existing["email"] == email for existing in self.users):
                self.rowcount = 0
                return
            self.users.append({"name": name, "email": email, "mlTrainingOptIn": ml_opt_in})
            self.rowcount = 1
            return

        raise AssertionError(f"Unexpected query: {query}")

    def fetchone(self):
        return self._last_row

    def close(self):
        return


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self):
        return

    def close(self):
        return


class SeedSyntheticUsersTests(unittest.TestCase):
    def setUp(self):
        self.cursor = FakeCursor()
        self.connection = FakeConnection(self.cursor)
        module.get_db_connection = lambda: self.connection
        module.SYNTHETIC_USER_EMAIL_PREFIX = "synthetic+"
        module.SYNTHETIC_USER_EMAIL_DOMAIN = "example.com"

    def test_seed_users_creates_opt_in_and_opt_out_targets(self):
        summary = module.seed_users(opt_in_target_count=5, opt_out_target_count=2)

        self.assertEqual(summary["opt_in"]["final"], 5)
        self.assertEqual(summary["opt_out"]["final"], 2)
        self.assertEqual(sum(1 for u in self.cursor.users if u["mlTrainingOptIn"]), 5)
        self.assertEqual(sum(1 for u in self.cursor.users if not u["mlTrainingOptIn"]), 2)

    def test_seed_users_is_idempotent(self):
        first = module.seed_users(opt_in_target_count=4, opt_out_target_count=3)
        second = module.seed_users(opt_in_target_count=4, opt_out_target_count=3)

        self.assertEqual(first["opt_in"]["created"], 4)
        self.assertEqual(first["opt_out"]["created"], 3)
        self.assertEqual(second["opt_in"]["created"], 0)
        self.assertEqual(second["opt_out"]["created"], 0)
        self.assertEqual(len(self.cursor.users), 7)

    def test_seed_users_only_fills_missing_per_cohort(self):
        self.cursor.users.append({"name": "existing-in", "email": "synthetic+existing-in@example.com", "mlTrainingOptIn": True})
        self.cursor.users.append({"name": "existing-out", "email": "synthetic+existing-out@example.com", "mlTrainingOptIn": False})

        summary = module.seed_users(opt_in_target_count=3, opt_out_target_count=2)

        self.assertEqual(summary["opt_in"]["existing"], 1)
        self.assertEqual(summary["opt_in"]["created"], 2)
        self.assertEqual(summary["opt_out"]["existing"], 1)
        self.assertEqual(summary["opt_out"]["created"], 1)
        self.assertEqual(sum(1 for u in self.cursor.users if u["mlTrainingOptIn"]), 3)
        self.assertEqual(sum(1 for u in self.cursor.users if not u["mlTrainingOptIn"]), 2)


if __name__ == "__main__":
    unittest.main()

import argparse
import os
import uuid

import psycopg2
from psycopg2.extras import RealDictCursor


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "database")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "immich")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
SYNTHETIC_USER_EMAIL_PREFIX = os.environ.get("SYNTHETIC_USER_EMAIL_PREFIX", "synthetic+")
SYNTHETIC_USER_EMAIL_DOMAIN = os.environ.get("SYNTHETIC_USER_EMAIL_DOMAIN", "example.com")
SYNTHETIC_USER_OPT_IN_COUNT = int(os.environ.get("SYNTHETIC_USER_OPT_IN_COUNT", "5000"))
SYNTHETIC_USER_OPT_OUT_COUNT = int(os.environ.get("SYNTHETIC_USER_OPT_OUT_COUNT", "1000"))


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def _count_existing(cur, ml_training_opt_in: bool) -> int:
    cur.execute(
        'SELECT count(*) AS count FROM "user" WHERE "email" LIKE %s AND "mlTrainingOptIn" = %s;',
        (f"{SYNTHETIC_USER_EMAIL_PREFIX}%", ml_training_opt_in),
    )
    return int(cur.fetchone()["count"])


def _create_users(cur, count: int, ml_training_opt_in: bool) -> int:
    created = 0
    for _ in range(max(0, count)):
        token = uuid.uuid4().hex
        email = f"{SYNTHETIC_USER_EMAIL_PREFIX}{token}@{SYNTHETIC_USER_EMAIL_DOMAIN}"
        cohort = "optin" if ml_training_opt_in else "optout"
        name = f"synthetic_{cohort}_{token[:8]}"
        cur.execute(
            '''
            INSERT INTO "user" ("name", "email", "mlTrainingOptIn")
            VALUES (%s, %s, %s)
            ON CONFLICT ("email") DO NOTHING;
            ''',
            (name, email, ml_training_opt_in),
        )
        if cur.rowcount > 0:
            created += 1

    return created


def seed_users(opt_in_target_count: int, opt_out_target_count: int) -> dict:
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    summary = {
        "opt_in": {"existing": 0, "created": 0, "final": 0, "target": max(0, int(opt_in_target_count))},
        "opt_out": {"existing": 0, "created": 0, "final": 0, "target": max(0, int(opt_out_target_count))},
    }

    try:
        opt_in_existing = _count_existing(cur, True)
        opt_in_needed = max(0, summary["opt_in"]["target"] - opt_in_existing)
        opt_in_created = _create_users(cur, opt_in_needed, True)

        opt_out_existing = _count_existing(cur, False)
        opt_out_needed = max(0, summary["opt_out"]["target"] - opt_out_existing)
        opt_out_created = _create_users(cur, opt_out_needed, False)

        summary["opt_in"]["existing"] = opt_in_existing
        summary["opt_in"]["created"] = opt_in_created
        summary["opt_in"]["final"] = opt_in_existing + opt_in_created

        summary["opt_out"]["existing"] = opt_out_existing
        summary["opt_out"]["created"] = opt_out_created
        summary["opt_out"]["final"] = opt_out_existing + opt_out_created

        conn.commit()
    finally:
        cur.close()
        conn.close()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic Immich users for sticker simulation")
    parser.add_argument("--opt-in-count", type=int, default=SYNTHETIC_USER_OPT_IN_COUNT, help="Target number of synthetic users with mlTrainingOptIn=true")
    parser.add_argument("--opt-out-count", type=int, default=SYNTHETIC_USER_OPT_OUT_COUNT, help="Target number of synthetic users with mlTrainingOptIn=false")
    args = parser.parse_args()

    summary = seed_users(args.opt_in_count, args.opt_out_count)
    print(
        "Synthetic users seeded. "
        f"prefix={SYNTHETIC_USER_EMAIL_PREFIX}, domain={SYNTHETIC_USER_EMAIL_DOMAIN}, "
        f"opt_in(target={summary['opt_in']['target']}, existing={summary['opt_in']['existing']}, created={summary['opt_in']['created']}, final={summary['opt_in']['final']}), "
        f"opt_out(target={summary['opt_out']['target']}, existing={summary['opt_out']['existing']}, created={summary['opt_out']['created']}, final={summary['opt_out']['final']})"
    )


if __name__ == "__main__":
    main()

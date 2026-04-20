import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List

import psycopg2
from pyspark.sql import SparkSession

from retraining_trigger_logic import should_trigger_retraining


POSTGRES_USER = os.environ.get("DB_USER")
POSTGRES_PASSWORD = os.environ.get("DB_PASS")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "5000"))
RETRAIN_COMMAND = os.environ.get("RETRAIN_COMMAND", "")


spark = (
    SparkSession.builder.appName("ML_Retraining_Trigger")
    .config(
        "spark.jars.packages",
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0,"
        "org.postgresql:postgresql:42.6.0,"
        "org.apache.hadoop:hadoop-aws:3.3.4,"
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.lakehouse", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.lakehouse.type", "hadoop")
    .config("spark.sql.catalog.lakehouse.warehouse", f"s3a://{RAW_BUCKET}/iceberg_warehouse")
    .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT)
    .config("spark.hadoop.fs.s3a.access.key", S3_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", S3_SECRET_KEY)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .getOrCreate()
)


def _write_table(df, table_name: str) -> None:
    if df is None or df.rdd.isEmpty():
        return
    if spark.catalog.tableExists(table_name):
        df.writeTo(table_name).append()
    else:
        df.writeTo(table_name).create()


def _db_connection():
    return psycopg2.connect(host="postgres", database="sticker_gen", user=POSTGRES_USER, password=POSTGRES_PASSWORD)


def _count_ready_rows(cur) -> int:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM sticker_generations sg
        JOIN users u ON sg.user_id = u.user_id
        WHERE sg.saved = TRUE
          AND sg.used_for_training = FALSE
          AND u.ml_training_opt_in = TRUE
          AND sg.quality_status = 'pass'
        """
    )
    return int(cur.fetchone()[0])


def _select_ready_rows(cur, target_count: int) -> List[Dict[str, Any]]:
    cur.execute(
        """
        SELECT
            sg.generation_id,
            sg.user_id,
            sg.image_id,
            sg.bbox::text AS bbox,
            sg.point_coords::text AS point_coords,
            sg.ml_suggested_mask,
            sg.user_saved_mask,
            sg.s3_sticker_key,
            sg.processing_time_ms,
            sg.num_tries,
            sg.edited_pixels,
            sg.generated_at,
            sg.quality_status
        FROM sticker_generations sg
        JOIN users u ON sg.user_id = u.user_id
        WHERE sg.saved = TRUE
          AND sg.used_for_training = FALSE
          AND u.ml_training_opt_in = TRUE
          AND sg.quality_status = 'pass'
        ORDER BY sg.generation_id ASC
        LIMIT %s
        """,
        (target_count,),
    )
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _run_retraining_command(retrain_run_id: str) -> bool:
    if not RETRAIN_COMMAND.strip():
        print("RETRAIN_COMMAND is not configured. Aborting without marking used_for_training.")
        return False

    command = RETRAIN_COMMAND.replace("{retrain_run_id}", retrain_run_id)
    print(f"Executing retraining command: {command}")
    completed = subprocess.run(command, shell=True, check=False)
    return completed.returncode == 0


def _mark_used_for_training(cur, generation_ids: List[int], retrain_run_id: str) -> None:
    if not generation_ids:
        return
    cur.execute(
        """
        UPDATE sticker_generations
        SET used_for_training = TRUE,
            used_for_training_at = %s,
            retrain_run_id = %s
        WHERE generation_id = ANY(%s)
        """,
        (datetime.utcnow(), retrain_run_id, generation_ids),
    )


def trigger_retraining(dry_run: bool = False) -> None:
    retrain_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    print(f"Retraining trigger run: {retrain_run_id}")
    print(f"Dry run: {dry_run}")
    print(f"Threshold: {RETRAIN_THRESHOLD}")

    conn = _db_connection()
    conn.autocommit = False
    cur = conn.cursor()

    selected_rows: List[Dict[str, Any]] = []
    ready_count = 0
    status = "skipped"
    error_message = ""

    try:
        cur.execute("SELECT pg_advisory_xact_lock(884422)")

        ready_count = _count_ready_rows(cur)
        if not should_trigger_retraining(ready_count, RETRAIN_THRESHOLD):
            print(f"Ready rows {ready_count} below threshold {RETRAIN_THRESHOLD}. No changes made.")
            status = "below_threshold"
        else:
            selected_rows = _select_ready_rows(cur, RETRAIN_THRESHOLD)
            if len(selected_rows) < RETRAIN_THRESHOLD:
                print("Ready count changed during selection; aborting safely.")
                status = "race_condition_abort"
            else:
                if dry_run:
                    status = "dry_run"
                else:
                    training_payload = []
                    selected_ids = []
                    selected_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                    for row in selected_rows:
                        out = dict(row)
                        out["retrain_run_id"] = retrain_run_id
                        out["selected_for_retraining_at"] = selected_at
                        training_payload.append(out)
                        selected_ids.append(int(row["generation_id"]))

                    _write_table(spark.createDataFrame(training_payload), "lakehouse.ml_datasets.training_data")

                    success = _run_retraining_command(retrain_run_id)
                    if success:
                        _mark_used_for_training(cur, selected_ids, retrain_run_id)
                        status = "succeeded"
                        print(f"Retraining succeeded and marked {len(selected_ids)} rows as used_for_training.")
                    else:
                        status = "failed"
                        error_message = "retraining_command_failed"
                        print("Retraining command failed. No used_for_training updates applied.")

        run_payload = [
            {
                "retrain_run_id": retrain_run_id,
                "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "status": status,
                "threshold": RETRAIN_THRESHOLD,
                "selected_count": len(selected_rows),
                "ready_count_at_start": ready_count,
                "error_message": error_message,
            }
        ]
        _write_table(spark.createDataFrame(run_payload), "lakehouse.ml_datasets.retraining_runs")

        if status in ("succeeded", "below_threshold", "dry_run", "race_condition_abort"):
            conn.commit()
        else:
            conn.rollback()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger retraining when enough QC-passed points are available")
    parser.add_argument("--dry-run", action="store_true", help="Check threshold and selection without retraining")
    args = parser.parse_args()
    trigger_retraining(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

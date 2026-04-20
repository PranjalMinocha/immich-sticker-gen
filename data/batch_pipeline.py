import argparse
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

from retraining_checks import load_quality_config, should_block_batch, summarize_validation, validate_rows


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "database")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "immich")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")
MAX_SAMPLES_PER_USER = int(os.environ.get("MAX_SAMPLES_PER_USER", "10"))
QC_SCAN_LIMIT = int(os.environ.get("QC_SCAN_LIMIT", "20000"))
QUALITY_CHECK_VERSION = int(os.environ.get("QUALITY_CHECK_VERSION", "1"))


def _jdbc_url() -> str:
    return os.environ.get("POSTGRES_URI") or f"jdbc:postgresql://{POSTGRES_HOST}:5432/{POSTGRES_DB}"


spark = (
    SparkSession.builder.appName("ML_Retraining_QC_Classifier")
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


def _extract_pending_qc_candidates():
    query = f"""
    (SELECT
        sg."id" AS "generationId",
        sg."userId" AS "userId",
        sg."assetId" AS "assetId",
        sg."bbox"::text AS "bbox",
        sg."pointCoords"::text AS "pointCoords",
        sg."mlSuggestedMask" AS "mlSuggestedMask",
        sg."userSavedMask" AS "userSavedMask",
        sg."s3StickerKey" AS "s3StickerKey",
        sg."processingTimeMs" AS "processingTimeMs",
        sg."numTries" AS "numTries",
        sg."editedPixels" AS "editedPixels",
        sg."createdAt" AS "createdAt",
        sg."saved" AS "saved",
        sg."usedForTraining" AS "usedForTraining",
        sg."qualityStatus" AS "qualityStatus",
        u."mlTrainingOptIn" AS "mlTrainingOptIn"
    FROM "sticker_generation" sg
    JOIN "user" u ON sg."userId" = u."id"
    WHERE sg."saved" = TRUE
      AND sg."usedForTraining" = FALSE
      AND u."mlTrainingOptIn" = TRUE
      AND COALESCE(sg."qualityStatus", 'pending') = 'pending'
    ORDER BY sg."createdAt", sg."id"
    LIMIT {QC_SCAN_LIMIT}) AS pending_qc_candidates
    """

    return (
        spark.read.format("jdbc")
        .option("url", _jdbc_url())
        .option("driver", "org.postgresql.Driver")
        .option("dbtable", query)
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .load()
    )


def _as_dict_rows(df) -> List[Dict[str, Any]]:
    return [row.asDict(recursive=True) for row in df.collect()]


def _update_qc_status(pass_entries: List[Dict[str, Any]], fail_entries: List[Dict[str, Any]], dry_run: bool) -> None:
    if dry_run:
        return

    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    cur = conn.cursor()
    now_ts = datetime.utcnow()
    try:
        if pass_entries:
            pass_ids = [entry["row"]["generationId"] for entry in pass_entries]
            cur.execute(
                """
                UPDATE "sticker_generation"
                SET "qualityStatus" = 'pass',
                    "qualityCheckedAt" = %s,
                    "qualityCheckVersion" = %s,
                    "qualityFailReasonsJson" = NULL
                WHERE "id" = ANY(%s::uuid[])
                """,
                (now_ts, QUALITY_CHECK_VERSION, pass_ids),
            )

        for entry in fail_entries:
            generation_id = entry["row"]["generationId"]
            fail_reasons_json = json.dumps(entry.get("hard_fail_reasons", []), sort_keys=True)
            cur.execute(
                """
                UPDATE "sticker_generation"
                SET "qualityStatus" = 'fail',
                    "qualityCheckedAt" = %s,
                    "qualityCheckVersion" = %s,
                    "qualityFailReasonsJson" = %s
                WHERE "id" = %s
                """,
                (now_ts, QUALITY_CHECK_VERSION, fail_reasons_json, generation_id),
            )

        conn.commit()
    finally:
        cur.close()
        conn.close()


def run_quality_classification(dry_run: bool = False) -> None:
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    quality_cfg = load_quality_config()

    print("Starting QC classification for retraining candidates...")
    print(f"Run ID: {run_id}")
    print(f"Dry run: {dry_run}")

    df_candidates = _extract_pending_qc_candidates()

    window_spec = Window.partitionBy("userId").orderBy(col("createdAt"), col("generationId"))
    df_filtered = (
        df_candidates.withColumn("user_sample_num", row_number().over(window_spec))
        .filter(col("user_sample_num") <= MAX_SAMPLES_PER_USER)
        .drop("user_sample_num")
        .orderBy(col("createdAt"), col("generationId"))
    )

    candidate_rows = _as_dict_rows(df_filtered)
    if not candidate_rows:
        print("No pending candidates need quality checks.")
        return

    validated = validate_rows(candidate_rows, quality_cfg)
    summary = summarize_validation(validated)
    summary["run_id"] = run_id
    summary["quality_check_version"] = QUALITY_CHECK_VERSION

    blocked, block_reasons = should_block_batch(summary, quality_cfg)
    summary["blocked"] = blocked
    summary["block_reasons"] = block_reasons

    pass_entries = [entry for entry in validated if entry["status"] in ("pass", "soft_warn")]
    fail_entries = [entry for entry in validated if entry["status"] == "hard_fail"]

    print(json.dumps(summary, indent=2, sort_keys=True))

    spark.sql("CREATE NAMESPACE IF NOT EXISTS lakehouse.ml_datasets")

    rejected_table = "lakehouse.ml_datasets.training_data_rejected"
    run_table = "lakehouse.ml_datasets.training_data_quality_runs"

    rejected_payload = []
    for entry in fail_entries:
        row = entry["row"]
        rejected_payload.append(
            {
                "run_id": run_id,
                "checked_at": entry["checked_at"],
                "quality_check_version": QUALITY_CHECK_VERSION,
                "generationId": row.get("generationId"),
                "userId": row.get("userId"),
                "assetId": row.get("assetId"),
                "hard_fail_reasons_json": json.dumps(entry.get("hard_fail_reasons", []), sort_keys=True),
                "soft_warn_reasons_json": json.dumps(entry.get("soft_warn_reasons", []), sort_keys=True),
                "metrics_json": json.dumps(entry.get("metrics", {}), sort_keys=True),
            }
        )

    run_payload = [
        {
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "quality_check_version": QUALITY_CHECK_VERSION,
            "total_candidates": int(summary["total_candidates"]),
            "accepted_count": int(summary["accepted_count"]),
            "hard_fail_count": int(summary["hard_fail_count"]),
            "soft_warn_count": int(summary["soft_warn_count"]),
            "hard_fail_rate": float(summary["hard_fail_rate"]),
            "soft_warn_rate": float(summary["soft_warn_rate"]),
            "blocked": bool(summary["blocked"]),
            "block_reasons_json": json.dumps(summary["block_reasons"], sort_keys=True),
            "top_hard_fail_reasons_json": json.dumps(summary["top_hard_fail_reasons"], sort_keys=True),
            "top_soft_warn_reasons_json": json.dumps(summary["top_soft_warn_reasons"], sort_keys=True),
        }
    ]

    _write_table(spark.createDataFrame(run_payload), run_table)
    if rejected_payload:
        _write_table(spark.createDataFrame(rejected_payload), rejected_table)

    if blocked:
        raise RuntimeError("QC batch failed quality gates: " + ", ".join(block_reasons))

    _update_qc_status(pass_entries, fail_entries, dry_run=dry_run)
    print(
        "QC classification complete. pass_or_warn={}, fail={}, version={}".format(
            len(pass_entries), len(fail_entries), QUALITY_CHECK_VERSION
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality checks and classify pending retraining candidates")
    parser.add_argument("--dry-run", action="store_true", help="Run checks and write summaries without DB status updates")
    args = parser.parse_args()
    run_quality_classification(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import shlex
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
import boto3
from pyspark.sql import SparkSession

from compile_retraining_dataset import compile_retraining_dataset
from model_deployer import deploy_model_from_mlflow_run, ping_serving_reload
from model_source_resolver import resolve_pretrained_model_source
from retraining_result_validation import validate_training_result
from retraining_trigger_logic import should_trigger_retraining
from rollback_monitor import record_deploy


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "database")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "immich")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "5000"))
RETRAIN_COMMAND = os.environ.get("RETRAIN_COMMAND", "")
RETRAIN_RESULT_DIR = os.environ.get("RETRAIN_RESULT_DIR", "/tmp/retraining_results")
RETRAIN_RUNS_TABLE = os.environ.get("RETRAIN_RUNS_TABLE", "lakehouse.ml_datasets.retraining_runs")
TRAINING_DATA_TABLE = os.environ.get("TRAINING_DATA_TABLE", "lakehouse.ml_datasets.training_data")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.27.60:8000")
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "immich-sticker-mobilesam")
MODEL_REGISTRY_ALIAS = os.environ.get("MODEL_REGISTRY_ALIAS", "Production")
BOOTSTRAP_MODEL_URI = os.environ.get("BOOTSTRAP_MODEL_URI", "")

VAL_MANIFEST_S3_URI = os.environ.get("VAL_MANIFEST_S3_URI", "")
DEPLOY_MODEL_AFTER_RETRAIN = os.environ.get("DEPLOY_MODEL_AFTER_RETRAIN", "false").lower() == "true"

# Docker-based training launch (used when TRAINING_DOCKER_IMAGE is set)
TRAINING_DOCKER_IMAGE       = os.environ.get("TRAINING_DOCKER_IMAGE", "")
TRAINING_BASE_CONFIG_PATH   = os.environ.get("TRAINING_BASE_CONFIG_PATH", "")
OBJECT_STORE_MOUNT_PATH     = os.environ.get("OBJECT_STORE_MOUNT_PATH", os.path.expanduser("~/my_object_store"))
TRAINING_NUM_WORKERS        = int(os.environ.get("TRAINING_NUM_WORKERS", "1"))
TRAINING_DOCKER_EXTRA_ARGS  = os.environ.get("TRAINING_DOCKER_EXTRA_ARGS", "")
MODEL_ARTIFACT_PATH = os.environ.get("MODEL_ARTIFACT_PATH", "checkpoints/mobile_sam_full.pt")
SERVING_MODEL_BUCKET = os.environ.get("SERVING_MODEL_BUCKET", RAW_BUCKET)
SERVING_MODEL_KEY = os.environ.get("SERVING_MODEL_KEY", "models/production/mobile_sam.pt")
PRETRAINED_MODEL_S3_URI = os.environ.get("PRETRAINED_MODEL_S3_URI", "")
SERVING_RELOAD_URL = os.environ.get("SERVING_RELOAD_URL", "")
SERVING_RELOAD_TOKEN = os.environ.get("SERVING_RELOAD_TOKEN", "")
DRY_RUN_SKIP_SPARK_LOG_WRITE = os.environ.get("DRY_RUN_SKIP_SPARK_LOG_WRITE", "true").lower() == "true"


_SPARK_SESSION: Optional[SparkSession] = None

def _get_spark() -> SparkSession:
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        return _SPARK_SESSION

    _SPARK_SESSION = (
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
    return _SPARK_SESSION


def _write_table(df, table_name: str) -> None:
    if df is None or df.rdd.isEmpty():
        return
    spark = _get_spark()
    if spark.catalog.tableExists(table_name):
        df.writeTo(table_name).append()
    else:
        df.writeTo(table_name).create()


def _db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _object_exists(s3_client, bucket: str, key: str) -> bool:
    if not key:
        return False
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _filter_rows_with_existing_objects(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    s3_client = _s3_client()
    filtered = []
    missing = 0
    for row in rows:
        key = row.get("originalPath")
        if _object_exists(s3_client, RAW_BUCKET, key):
            filtered.append(row)
        else:
            missing += 1
    if missing:
        print(f"Skipped {missing} candidate rows with missing sticker objects")
    return filtered


def _count_ready_rows(cur) -> int:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM "sticker_generation" sg
        JOIN "user" u ON sg."userId" = u."id"
        WHERE sg."saved" = TRUE
          AND sg."usedForTraining" = FALSE
          AND u."mlTrainingOptIn" = TRUE
          AND sg."qualityStatus" = 'pass'
        """
    )
    return int(cur.fetchone()[0])


def _select_ready_rows(cur, target_count: int) -> List[Dict[str, Any]]:
    cur.execute(
        """
        SELECT
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
            sg."qualityStatus" AS "qualityStatus",
            a."originalPath" AS "originalPath"
        FROM "sticker_generation" sg
        JOIN "user" u ON sg."userId" = u."id"
        JOIN "asset" a ON sg."assetId" = a."id"
        WHERE sg."saved" = TRUE
          AND sg."usedForTraining" = FALSE
          AND u."mlTrainingOptIn" = TRUE
          AND sg."qualityStatus" = 'pass'
        ORDER BY sg."createdAt" ASC, sg."id" ASC
        LIMIT %s
        """,
        (target_count,),
    )
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _json_safe_str(value: str) -> str:
    return shlex.quote(value)


def _default_retrain_command() -> str:
    script = os.environ.get("TRAINING_SCRIPT_PATH")
    config = os.environ.get("TRAINING_CONFIG_PATH")
    workers = os.environ.get("TRAINING_NUM_WORKERS", "1")
    if not script or not config:
        return ""
    return (
        f"python3 {_json_safe_str(script)} "
        f"--config {_json_safe_str(config)} "
        f"--num-workers {_json_safe_str(workers)} "
        f"--run-id {{retrain_run_id}} "
        f"--output-json {{result_json_path}}"
    )


def _resolve_pretrained_model_source_uri() -> tuple[str, str, str]:
    object_store_model_uri = PRETRAINED_MODEL_S3_URI.strip()
    if not object_store_model_uri and SERVING_MODEL_BUCKET and SERVING_MODEL_KEY:
        object_store_model_uri = f"s3://{SERVING_MODEL_BUCKET}/{SERVING_MODEL_KEY}"

    resolved = resolve_pretrained_model_source(
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=MODEL_REGISTRY_NAME,
        preferred_alias=MODEL_REGISTRY_ALIAS,
        bootstrap_model_uri=BOOTSTRAP_MODEL_URI or None,
        object_store_model_uri=object_store_model_uri or None,
        object_store_client=_s3_client(),
    )
    print(
        "Resolved pretrained model source: uri={} strategy={} model={} version={}".format(
            resolved.source_uri,
            resolved.strategy,
            resolved.model_name,
            resolved.model_version or "n/a",
        )
    )
    return resolved.source_uri, resolved.strategy, resolved.model_version or ""


def _s3_uri_to_container_path(s3_uri: str, container_root: str = "/data") -> str:
    """Convert s3://{RAW_BUCKET}/key → {container_root}/key."""
    prefix = f"s3://{RAW_BUCKET}/"
    if s3_uri.startswith(prefix):
        return container_root.rstrip("/") + "/" + s3_uri[len(prefix):]
    raise ValueError(f"S3 URI does not start with expected bucket prefix: {s3_uri}")


def _download_pretrained_for_docker(pretrained_model_uri: str, dest_path: str) -> None:
    """Download the pretrained model to dest_path for bind-mount into the container."""
    import tempfile
    if pretrained_model_uri.startswith("s3://"):
        stripped = pretrained_model_uri[5:]
        bucket, _, key = stripped.partition("/")
        _s3_client().download_file(bucket, key, dest_path)
    elif pretrained_model_uri.startswith(("models:/", "runs:/", "mlflow-artifacts:/")):
        import mlflow.artifacts as mlflow_artifacts
        tmp = tempfile.mkdtemp(prefix="pretrained_dl_")
        downloaded = mlflow_artifacts.download_artifacts(
            artifact_uri=pretrained_model_uri,
            tracking_uri=MLFLOW_TRACKING_URI,
            dst_path=tmp,
        )
        src = Path(downloaded)
        if src.is_dir():
            pts = list(src.rglob("*.pt"))
            if not pts:
                raise RuntimeError(f"No .pt file in downloaded artifact dir: {src}")
            src = pts[0]
        import shutil
        shutil.move(str(src), dest_path)
    else:
        raise ValueError(f"Cannot download pretrained model from URI: {pretrained_model_uri}")


def _build_docker_retrain_command(
    retrain_run_id: str,
    train_manifest_s3_uri: str,
    val_manifest_s3_uri: str,
    pretrained_model_uri: str,
    result_json_path: str,
) -> str:
    """
    Build a `docker run` command that launches the training container.

    Layout inside the container:
      /data   ← rclone FUSE mount (OBJECT_STORE_MOUNT_PATH) — all S3 data accessible here
      /out    ← RETRAIN_RESULT_DIR bind-mount — result JSON written here
      /config.yaml ← generated per-run YAML config
    """
    import tempfile, yaml  # yaml is always available in the training image

    result_dir = str(RETRAIN_RESULT_DIR)
    os.makedirs(result_dir, exist_ok=True)

    # Convert S3 manifest URIs to container-local paths (via rclone mount at /data)
    train_manifest_container = _s3_uri_to_container_path(train_manifest_s3_uri)
    val_manifest_container   = _s3_uri_to_container_path(val_manifest_s3_uri)

    # Download pretrained model to result_dir so it's accessible in the container
    pretrained_local = os.path.join(result_dir, f"{retrain_run_id}_pretrained.pt")
    _download_pretrained_for_docker(pretrained_model_uri, pretrained_local)

    # Build per-run config by loading the base config and overriding run-specific fields
    base_cfg: Dict[str, Any] = {}
    if TRAINING_BASE_CONFIG_PATH and os.path.isfile(TRAINING_BASE_CONFIG_PATH):
        with open(TRAINING_BASE_CONFIG_PATH, "r") as f:
            base_cfg = yaml.safe_load(f) or {}

    run_cfg = dict(base_cfg)
    run_cfg.setdefault("training", {})
    run_cfg["training"]["use_pretrained"] = True
    run_cfg["training"]["pretrained_checkpoint_path"] = f"/out/{retrain_run_id}_pretrained.pt"
    run_cfg.setdefault("data", {})
    run_cfg["data"]["train_manifest_csv"] = train_manifest_container
    run_cfg["data"]["val_manifest_csv"]   = val_manifest_container
    run_cfg["data"]["objstore_local_root"] = "/data"
    run_cfg.setdefault("mlflow", {})
    run_cfg["mlflow"]["tracking_uri"] = MLFLOW_TRACKING_URI
    run_cfg.setdefault("output", {})
    run_cfg["output"]["dir"] = "/out"
    if TRAINING_BASE_CONFIG_PATH:
        run_cfg["mobilesam_root"] = base_cfg.get("mobilesam_root", "/app/MobileSAM")

    config_path = os.path.join(result_dir, f"{retrain_run_id}_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(run_cfg, f)

    # result_json_path is on the host; inside the container it's at /out/<basename>
    result_json_container = "/out/" + Path(result_json_path).name

    env_args = " ".join([
        f"-e MLFLOW_TRACKING_URI={shlex.quote(MLFLOW_TRACKING_URI)}",
        f"-e S3_ENDPOINT={shlex.quote(S3_ENDPOINT or '')}",
        f"-e S3_ACCESS_KEY={shlex.quote(S3_ACCESS_KEY or '')}",
        f"-e S3_SECRET_KEY={shlex.quote(S3_SECRET_KEY or '')}",
        f"-e RAW_BUCKET={shlex.quote(RAW_BUCKET or '')}",
    ])

    volume_args = " ".join([
        f"-v {shlex.quote(os.path.abspath(OBJECT_STORE_MOUNT_PATH))}:/data:ro",
        f"-v {shlex.quote(os.path.abspath(result_dir))}:/out",
        f"-v {shlex.quote(os.path.abspath(config_path))}:/config.yaml:ro",
    ])

    gpu_arg = "--gpus all"
    extra = TRAINING_DOCKER_EXTRA_ARGS.strip()

    cmd = (
        f"docker run --rm {gpu_arg} {volume_args} {env_args} {extra} "
        f"{shlex.quote(TRAINING_DOCKER_IMAGE)} "
        f"python train.py --config /config.yaml "
        f"--num-workers {TRAINING_NUM_WORKERS} "
        f"--run-id {shlex.quote(retrain_run_id)} "
        f"--output-json {shlex.quote(result_json_container)}"
    )
    return cmd


def _run_command(command: str, cwd: Optional[str] = None) -> bool:
    completed = subprocess.run(command, shell=True, check=False, cwd=cwd)
    return completed.returncode == 0


def _parse_training_result(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Training result json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training result payload must be an object")
    return payload


def _upload_json_to_s3(s3_uri_key: str, payload: Dict[str, Any]) -> str:
    s3_client = _s3_client()
    s3_client.put_object(
        Bucket=RAW_BUCKET,
        Key=s3_uri_key,
        Body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{RAW_BUCKET}/{s3_uri_key}"


def _deploy_model_if_enabled(training_result: Dict[str, Any]) -> tuple[bool, str, str]:
    if not DEPLOY_MODEL_AFTER_RETRAIN:
        return False, "deploy_disabled", ""

    mlflow_payload = training_result.get("mlflow", {}) if isinstance(training_result, dict) else {}
    run_id = mlflow_payload.get("runId")
    tracking_uri = mlflow_payload.get("trackingUri") or MLFLOW_TRACKING_URI
    if not run_id:
        return False, "missing_mlflow_run_id", ""

    deploy_local_dir = os.path.join(RETRAIN_RESULT_DIR, "deploy_cache")
    os.makedirs(deploy_local_dir, exist_ok=True)

    deployed = deploy_model_from_mlflow_run(
        s3_client=_s3_client(),
        raw_bucket=RAW_BUCKET,
        tracking_uri=tracking_uri,
        run_id=str(run_id),
        model_artifact_path=MODEL_ARTIFACT_PATH,
        serving_model_bucket=SERVING_MODEL_BUCKET,
        serving_model_key=SERVING_MODEL_KEY,
        local_dir=deploy_local_dir,
        backup_model_key=os.environ.get("BACKUP_MODEL_KEY", SERVING_MODEL_KEY + ".backup"),
    )

    reload_ok, reload_detail = ping_serving_reload(SERVING_RELOAD_URL, SERVING_RELOAD_TOKEN or None)
    if SERVING_RELOAD_URL and not reload_ok:
        return False, f"reload_failed:{reload_detail}", deployed["target_s3_uri"]

    return True, reload_detail if reload_ok else "reloaded_not_requested", deployed["target_s3_uri"]


def _execute_retraining(
    retrain_run_id: str,
    train_manifest_s3_uri: str,
    val_manifest_s3_uri: str,
    metadata_s3_uri: str,
    pretrained_model_uri: str,
) -> tuple[bool, str, str, Dict[str, Any], str]:
    os.makedirs(RETRAIN_RESULT_DIR, exist_ok=True)
    result_json_path = Path(RETRAIN_RESULT_DIR) / f"{retrain_run_id}_result.json"

    if TRAINING_DOCKER_IMAGE:
        try:
            command = _build_docker_retrain_command(
                retrain_run_id=retrain_run_id,
                train_manifest_s3_uri=train_manifest_s3_uri,
                val_manifest_s3_uri=val_manifest_s3_uri,
                pretrained_model_uri=pretrained_model_uri,
                result_json_path=str(result_json_path),
            )
        except Exception as exc:
            return False, f"docker_command_build_failed:{exc}", str(result_json_path), {}, ""
    else:
        command_template = RETRAIN_COMMAND.strip() or _default_retrain_command()
        if not command_template:
            print("RETRAIN_COMMAND is not configured and TRAINING_DOCKER_IMAGE is not set.")
            return False, "retraining_command_missing", "", {}, ""
        command = command_template.format(
            retrain_run_id=retrain_run_id,
            train_manifest_s3_uri=train_manifest_s3_uri,
            val_manifest_s3_uri=val_manifest_s3_uri,
            metadata_s3_uri=metadata_s3_uri,
            pretrained_model_uri=pretrained_model_uri,
            result_json_path=str(result_json_path),
        )

    print(f"Executing retraining command: {command}")
    success = _run_command(command, cwd=os.environ.get("TRAINING_WORKDIR"))
    if not success:
        return False, "retraining_command_failed", str(result_json_path), {}, ""

    payload = _parse_training_result(result_json_path)
    valid, reason = validate_training_result(payload)
    if not valid:
        return False, reason, str(result_json_path), payload, ""

    return True, "", str(result_json_path), payload, payload["mlflow"]["runId"]


def _mark_used_for_training(cur, generation_ids: List[str], retrain_run_id: str) -> None:
    if not generation_ids:
        return
    cur.execute(
        """
        UPDATE "sticker_generation"
        SET "usedForTraining" = TRUE,
            "usedForTrainingAt" = %s,
            "retrainRunId" = %s
        WHERE "id" = ANY(%s::uuid[])
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
    training_result: Dict[str, Any] = {}
    training_result_s3_uri = ""
    mlflow_run_id = ""
    model_name = ""
    model_version = ""
    quality_gate_passed = False
    train_manifest_s3_uri = ""
    val_manifest_s3_uri = ""
    retrain_metadata_s3_uri = ""
    pretrained_model_uri = ""
    pretrained_model_strategy = ""
    pretrained_model_version = ""
    deployment_status = "skipped"
    deployment_detail = ""
    deployed_model_s3_uri = ""

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
                    selected_rows = _filter_rows_with_existing_objects(selected_rows)
                    if len(selected_rows) < RETRAIN_THRESHOLD:
                        print("Ready rows dropped below threshold after object-existence checks; aborting safely.")
                        status = "object_missing_abort"
                    else:
                        try:
                            compiled = compile_retraining_dataset(
                                selected_rows,
                                retrain_run_id,
                                static_val_manifest_s3_uri=VAL_MANIFEST_S3_URI or None,
                            )
                        except Exception as exc:
                            status = "dataset_compile_failed"
                            error_message = str(exc)
                            print(f"Dataset compile failed: {exc}")
                            compiled = None

                        if compiled is not None:
                            success = False
                            reason = "retraining_not_started"
                            result_json_path = ""
                            selected_ids = list(compiled.accepted_generation_ids)
                            train_manifest_s3_uri = compiled.train_manifest_s3_uri
                            val_manifest_s3_uri = compiled.val_manifest_s3_uri
                            retrain_metadata_s3_uri = compiled.metadata_s3_uri
                            try:
                                (
                                    pretrained_model_uri,
                                    pretrained_model_strategy,
                                    pretrained_model_version,
                                ) = _resolve_pretrained_model_source_uri()
                            except Exception as exc:
                                success = False
                                reason = f"pretrained_source_resolve_failed:{exc}"
                                result_json_path = ""
                                training_result = {}
                                mlflow_run_id = ""
                                compiled = None
                            try:
                                if compiled is not None:
                                    success, reason, result_json_path, training_result, mlflow_run_id = _execute_retraining(
                                        retrain_run_id=retrain_run_id,
                                        train_manifest_s3_uri=compiled.train_manifest_s3_uri,
                                        val_manifest_s3_uri=compiled.val_manifest_s3_uri,
                                        metadata_s3_uri=compiled.metadata_s3_uri,
                                        pretrained_model_uri=pretrained_model_uri,
                                    )
                            except Exception as exc:
                                success = False
                                reason = f"retraining_runtime_error:{exc}"
                                result_json_path = ""
                                training_result = {}
                                mlflow_run_id = ""
                        else:
                            success = False
                            reason = "dataset_compile_failed"
                            result_json_path = ""

                        if success:
                            training_payload = []
                            selected_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                            selected_id_set = set(selected_ids)
                            for row in selected_rows:
                                if row["generationId"] not in selected_id_set:
                                    continue
                                out = dict(row)
                                out["retrainRunId"] = retrain_run_id
                                out["selectedForRetrainingAt"] = selected_at
                                training_payload.append(out)

                            _write_table(_get_spark().createDataFrame(training_payload), TRAINING_DATA_TABLE)

                            if training_result:
                                mlflow_payload = training_result.get("mlflow", {})
                                model_name = str(mlflow_payload.get("modelName", ""))
                                model_version = str(mlflow_payload.get("modelVersion", ""))
                                quality_gate_passed = bool(training_result.get("qualityGate", {}).get("passed", False))
                                result_key = f"retraining_runs/{retrain_run_id}/training_result.json"
                                training_result_s3_uri = _upload_json_to_s3(result_key, training_result)

                            allow_mark = True
                            try:
                                deployed, deployment_detail, deployed_model_s3_uri = _deploy_model_if_enabled(training_result)
                                deployment_status = "succeeded" if deployed else (
                                    "skipped" if deployment_detail in ("deploy_disabled", "reloaded_not_requested") else "failed"
                                )
                                if deployment_status == "failed":
                                    status = "failed"
                                    error_message = f"deployment_failed:{deployment_detail}"
                                    allow_mark = False
                                    print("Deployment failed. No usedForTraining updates applied.")
                            except Exception as exc:
                                status = "failed"
                                error_message = f"deployment_failed:{exc}"
                                deployment_status = "failed"
                                allow_mark = False
                                print("Deployment failed. No usedForTraining updates applied.")

                            if allow_mark:
                                _mark_used_for_training(cur, selected_ids, retrain_run_id)
                                status = "succeeded"
                                print(f"Retraining succeeded and marked {len(selected_ids)} rows as usedForTraining.")
                                if deployment_status == "succeeded":
                                    try:
                                        backup_key = os.environ.get(
                                            "BACKUP_MODEL_KEY", SERVING_MODEL_KEY + ".backup"
                                        )
                                        record_deploy(
                                            current_model_s3_key=SERVING_MODEL_KEY,
                                            previous_model_s3_key=backup_key,
                                        )
                                    except Exception as exc:
                                        print(f"[retraining_trigger] record_deploy failed (non-fatal): {exc}")
                        else:
                            status = "failed"
                            error_message = reason
                            print("Retraining command failed. No usedForTraining updates applied.")

        should_write_run_log = not (dry_run and DRY_RUN_SKIP_SPARK_LOG_WRITE)
        if should_write_run_log:
            run_payload = [
                {
                    "retrain_run_id": retrain_run_id,
                    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "status": status,
                    "threshold": RETRAIN_THRESHOLD,
                    "selected_count": len(selected_rows),
                    "ready_count_at_start": ready_count,
                    "error_message": error_message,
                    "mlflow_run_id": mlflow_run_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "quality_gate_passed": quality_gate_passed,
                    "training_result_s3_uri": training_result_s3_uri,
                    "train_manifest_s3_uri": train_manifest_s3_uri,
                    "val_manifest_s3_uri": val_manifest_s3_uri,
                    "retrain_metadata_s3_uri": retrain_metadata_s3_uri,
                    "pretrained_model_uri": pretrained_model_uri,
                    "pretrained_model_strategy": pretrained_model_strategy,
                    "pretrained_model_version": pretrained_model_version,
                    "deployment_status": deployment_status,
                    "deployment_detail": deployment_detail,
                    "deployed_model_s3_uri": deployed_model_s3_uri,
                }
            ]
            _write_table(_get_spark().createDataFrame(run_payload), RETRAIN_RUNS_TABLE)
        else:
            print("Dry run: skipping Spark/Iceberg run log write (DRY_RUN_SKIP_SPARK_LOG_WRITE=true).")

        if status in ("succeeded", "below_threshold", "dry_run", "race_condition_abort", "object_missing_abort"):
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

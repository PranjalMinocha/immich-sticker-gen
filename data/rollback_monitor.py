"""
Automated rollback monitor for the sticker-gen serving endpoint.

Checks three rollback triggers (after a 30-minute post-deploy warmup):
  1. Serving error rate > ERROR_RATE_THRESHOLD  (default 5%)
  2. Median IoU drops > IOU_DROP_THRESHOLD below the baseline  (default 15 pp)
  3. editedPixels p75 spikes > EDIT_PIXELS_SPIKE_FACTOR × baseline  (default 2×)

Run periodically (e.g. every 5 minutes via cron or a loop).  State is
persisted in ROLLBACK_STATE_PATH so the monitor survives restarts.

State file schema:
  {
    "deployed_at": "<ISO-8601 UTC>",
    "current_model_s3_key": "models/production/mobile_sam.pt",
    "previous_model_s3_key": "models/backup/mobile_sam.pt",
    "baseline_iou_median": 0.85,
    "baseline_edit_pixels_p75": 500,
    "rollback_count": 0
  }
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import psycopg2
import requests


# ── Environment config ────────────────────────────────────────────────────────
ROLLBACK_STATE_PATH       = os.environ.get("ROLLBACK_STATE_PATH", "/tmp/sticker_rollback_state.json")
PROMETHEUS_URL            = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
SERVING_URL               = os.environ.get("SERVING_URL", "http://localhost:8004")
SERVING_RELOAD_URL        = os.environ.get("SERVING_RELOAD_URL", "")
SERVING_RELOAD_TOKEN      = os.environ.get("SERVING_RELOAD_TOKEN", "")

POSTGRES_HOST             = os.environ.get("POSTGRES_HOST", "database")
POSTGRES_DB               = os.environ.get("POSTGRES_DB", "immich")
POSTGRES_USER             = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD         = os.environ.get("POSTGRES_PASSWORD", "postgres")

S3_ENDPOINT               = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY             = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY             = os.environ.get("S3_SECRET_KEY")
SERVING_MODEL_BUCKET      = os.environ.get("SERVING_MODEL_BUCKET", "objstore-proj28")
SERVING_MODEL_KEY         = os.environ.get("SERVING_MODEL_KEY", "models/production/mobile_sam.pt")
BACKUP_MODEL_KEY          = os.environ.get("BACKUP_MODEL_KEY", "models/backup/mobile_sam.pt")

MLFLOW_TRACKING_URI       = os.environ.get("MLFLOW_TRACKING_URI", "")
MODEL_NAME                = os.environ.get("MODEL_NAME", "immich-sticker-mobilesam")
MODEL_ARTIFACT_PATH       = os.environ.get("MODEL_ARTIFACT_PATH", "checkpoints/mobile_sam_full.pt")

WARMUP_MINUTES            = int(os.environ.get("WARMUP_MINUTES", "30"))
ERROR_RATE_THRESHOLD      = float(os.environ.get("ERROR_RATE_THRESHOLD", "0.05"))
IOU_DROP_THRESHOLD        = float(os.environ.get("IOU_DROP_THRESHOLD", "0.15"))
EDIT_PIXELS_SPIKE_FACTOR  = float(os.environ.get("EDIT_PIXELS_SPIKE_FACTOR", "2.0"))
METRICS_WINDOW_MINUTES    = int(os.environ.get("METRICS_WINDOW_MINUTES", "10"))
EDIT_PIXELS_SAMPLE_ROWS   = int(os.environ.get("EDIT_PIXELS_SAMPLE_ROWS", "200"))


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_state() -> Dict[str, Any]:
    try:
        with open(ROLLBACK_STATE_PATH, "r", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except FileNotFoundError:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    # Open for read+write (create if missing), hold exclusive lock for the full write.
    fd = os.open(ROLLBACK_STATE_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            f.truncate()
            json.dump(state, f, indent=2, sort_keys=True)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _minutes_since_deploy(state: Dict[str, Any]) -> float:
    deployed_at = state.get("deployed_at")
    if not deployed_at:
        return float("inf")
    dt = datetime.fromisoformat(deployed_at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


# ── Prometheus helpers ────────────────────────────────────────────────────────

def _prom_query(query: str) -> Optional[float]:
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            return None
        return float(results[0]["value"][1])
    except Exception as exc:
        print(f"[rollback_monitor] Prometheus query failed: {exc}")
        return None


def _get_error_rate(window_minutes: int) -> Optional[float]:
    window = f"{window_minutes}m"
    query = (
        f"sum(rate(sticker_errors_total[{window}])) / "
        f"(sum(rate(sticker_requests_total[{window}])) + 1e-9)"
    )
    return _prom_query(query)


def _get_iou_median(window_minutes: int) -> Optional[float]:
    window = f"{window_minutes}m"
    query = (
        f"histogram_quantile(0.5, "
        f"sum(rate(sticker_iou_score_bucket[{window}])) by (le))"
    )
    return _prom_query(query)


# ── Postgres helpers ──────────────────────────────────────────────────────────

def _get_edit_pixels_p75(sample_rows: int) -> Optional[float]:
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cur = conn.cursor()
        cur.execute(
            """
            SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "editedPixels")
            FROM (
                SELECT "editedPixels"
                FROM "sticker_generation"
                WHERE "saved" = TRUE
                ORDER BY "createdAt" DESC
                LIMIT %s
            ) recent
            """,
            (sample_rows,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0] is not None:
            return float(row[0])
        return None
    except Exception as exc:
        print(f"[rollback_monitor] Postgres query failed: {exc}")
        return None


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def _copy_model(src_key: str, dst_key: str) -> None:
    s3 = _s3_client()
    s3.copy_object(
        Bucket=SERVING_MODEL_BUCKET,
        CopySource={"Bucket": SERVING_MODEL_BUCKET, "Key": src_key},
        Key=dst_key,
    )


def _ping_reload() -> tuple[bool, str]:
    if not SERVING_RELOAD_URL:
        return False, "reload_url_unset"
    headers = {}
    if SERVING_RELOAD_TOKEN:
        headers["X-Model-Reload-Token"] = SERVING_RELOAD_TOKEN
    try:
        resp = requests.post(SERVING_RELOAD_URL, headers=headers, timeout=20)
        ok = resp.status_code < 300
        return ok, resp.text[:200]
    except Exception as exc:
        return False, str(exc)


# ── Force rollback ───────────────────────────────────────────────────────────

def force_rollback(to_version: Optional[int] = None, reason: str = "manual") -> dict:
    import tempfile
    from mlflow.tracking import MlflowClient
    from model_deployer import deploy_model_from_mlflow_run

    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI env var is required for force-rollback")

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, "Production")
        current_version = int(prod_mv.version)
    except Exception as exc:
        raise RuntimeError(f"Could not resolve Production alias for '{MODEL_NAME}': {exc}")

    target_version = to_version if to_version is not None else current_version - 1
    if target_version < 1:
        raise RuntimeError(
            f"Cannot roll back: current Production is version {current_version} "
            f"and there is no version {target_version}"
        )
    if target_version == current_version:
        raise RuntimeError(f"Version {target_version} is already Production — nothing to do")

    target_mv = client.get_model_version(MODEL_NAME, str(target_version))
    run_id = target_mv.run_id
    print(
        f"[rollback_monitor] force-rollback: version {current_version} → {target_version} "
        f"(run_id={run_id}, reason={reason!r})"
    )

    s3 = _s3_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        deploy_result = deploy_model_from_mlflow_run(
            s3_client=s3,
            raw_bucket=SERVING_MODEL_BUCKET,
            tracking_uri=MLFLOW_TRACKING_URI,
            run_id=run_id,
            model_artifact_path=MODEL_ARTIFACT_PATH,
            serving_model_bucket=SERVING_MODEL_BUCKET,
            serving_model_key=SERVING_MODEL_KEY,
            local_dir=tmpdir,
            backup_model_key=BACKUP_MODEL_KEY,
        )

    reload_ok, reload_detail = _ping_reload()

    client.set_registered_model_alias(MODEL_NAME, "Production", str(target_version))

    state = _load_state()
    state["deployed_at"] = _now_utc_iso()
    state["current_model_s3_key"] = SERVING_MODEL_KEY
    state["previous_model_s3_key"] = BACKUP_MODEL_KEY
    state["rollback_count"] = state.get("rollback_count", 0) + 1
    state["last_rollback_at"] = _now_utc_iso()
    state["last_rollback_reasons"] = [reason]
    _save_state(state)

    result = {
        "action": "force_rolled_back",
        "from_version": current_version,
        "to_version": target_version,
        "run_id": run_id,
        "reload_ok": reload_ok,
        "reload_detail": reload_detail,
        **deploy_result,
    }
    print(f"[rollback_monitor] force-rollback complete: {result}")
    return result


# ── Main check ────────────────────────────────────────────────────────────────

def check_and_rollback(dry_run: bool = False) -> Dict[str, Any]:
    state = _load_state()
    if not state:
        print("[rollback_monitor] No deployment state found. Run record-deploy first.")
        return {"action": "no_state"}

    minutes_up = _minutes_since_deploy(state)
    if minutes_up < WARMUP_MINUTES:
        remaining = WARMUP_MINUTES - minutes_up
        print(f"[rollback_monitor] In warmup window — {remaining:.1f} min remaining. No checks.")
        return {"action": "warmup", "remaining_minutes": remaining}

    baseline_iou       = state.get("baseline_iou_median")
    baseline_edit_p75  = state.get("baseline_edit_pixels_p75")

    error_rate   = _get_error_rate(METRICS_WINDOW_MINUTES)
    current_iou  = _get_iou_median(METRICS_WINDOW_MINUTES)
    edit_p75     = _get_edit_pixels_p75(EDIT_PIXELS_SAMPLE_ROWS)

    if error_rate is None and current_iou is None and edit_p75 is None:
        print(
            "[rollback_monitor] WARNING: all metric queries returned None — "
            "cannot assess model health. Check Prometheus and Postgres connectivity."
        )

    triggers: list[str] = []

    if error_rate is not None and error_rate > ERROR_RATE_THRESHOLD:
        triggers.append(f"error_rate={error_rate:.3f} > threshold={ERROR_RATE_THRESHOLD}")

    if (
        current_iou is not None
        and baseline_iou is not None
        and (baseline_iou - current_iou) > IOU_DROP_THRESHOLD
    ):
        triggers.append(
            f"iou_drop={baseline_iou - current_iou:.3f} > threshold={IOU_DROP_THRESHOLD} "
            f"(current={current_iou:.3f}, baseline={baseline_iou:.3f})"
        )

    if (
        edit_p75 is not None
        and baseline_edit_p75 is not None
        and baseline_edit_p75 > 0
        and edit_p75 > EDIT_PIXELS_SPIKE_FACTOR * baseline_edit_p75
    ):
        triggers.append(
            f"edit_pixels_p75={edit_p75:.0f} > {EDIT_PIXELS_SPIKE_FACTOR}x baseline={baseline_edit_p75:.0f}"
        )

    result: Dict[str, Any] = {
        "checked_at": _now_utc_iso(),
        "minutes_since_deploy": round(minutes_up, 1),
        "error_rate": error_rate,
        "current_iou": current_iou,
        "edit_pixels_p75": edit_p75,
        "baseline_iou_median": baseline_iou,
        "baseline_edit_pixels_p75": baseline_edit_p75,
        "triggers": triggers,
    }

    if not triggers:
        result["action"] = "ok"
        print(f"[rollback_monitor] All checks passed — error_rate={error_rate}, iou={current_iou}, edit_p75={edit_p75}")
        return result

    print(f"[rollback_monitor] Rollback triggers: {triggers}")

    if dry_run:
        result["action"] = "dry_run_would_rollback"
        return result

    previous_key = state.get("previous_model_s3_key")
    if not previous_key:
        result["action"] = "rollback_skipped_no_previous"
        print("[rollback_monitor] No previous model key in state — cannot roll back.")
        return result

    print(f"[rollback_monitor] Rolling back: {previous_key} → {SERVING_MODEL_KEY}")
    _copy_model(previous_key, SERVING_MODEL_KEY)

    reload_ok, reload_detail = _ping_reload()
    if not reload_ok:
        result["action"] = "rollback_reload_failed"
        result["reload_detail"] = reload_detail
        print(f"[rollback_monitor] Rollback copy succeeded but reload failed: {reload_detail}")
        return result

    state["rollback_count"] = state.get("rollback_count", 0) + 1
    state["last_rollback_at"] = _now_utc_iso()
    state["last_rollback_reasons"] = triggers
    # Swap current ↔ previous so a second rollback goes forward again
    state["previous_model_s3_key"] = state.get("current_model_s3_key", SERVING_MODEL_KEY)
    state["current_model_s3_key"]  = previous_key
    _save_state(state)

    result["action"] = "rolled_back"
    result["reload_detail"] = reload_detail
    print(f"[rollback_monitor] Rollback complete. reload_detail={reload_detail}")
    return result


def sample_baseline_metrics() -> Tuple[Optional[float], Optional[float]]:
    """
    Sample IoU median and editedPixels p75 from the CURRENT (pre-deploy) model.
    Call this BEFORE deploying a new model so the baseline is not contaminated
    by the new model's traffic.
    Returns (baseline_iou_median, baseline_edit_pixels_p75).
    """
    iou = _get_iou_median(METRICS_WINDOW_MINUTES)
    edit_p75 = _get_edit_pixels_p75(EDIT_PIXELS_SAMPLE_ROWS)
    print(f"[rollback_monitor] Pre-deploy baseline sampled: iou={iou}, edit_pixels_p75={edit_p75}")
    return iou, edit_p75


def record_deploy(
    current_model_s3_key: str,
    previous_model_s3_key: str,
    baseline_iou_median: Optional[float] = None,
    baseline_edit_pixels_p75: Optional[float] = None,
) -> None:
    """
    Call this immediately after a successful model deployment.
    Resets the warmup clock and records the baseline metrics.
    Prefer passing pre-sampled baseline values (captured before deploy) rather
    than letting this function sample them post-deploy.
    """
    if baseline_edit_pixels_p75 is None:
        baseline_edit_pixels_p75 = _get_edit_pixels_p75(EDIT_PIXELS_SAMPLE_ROWS)
    if baseline_iou_median is None:
        baseline_iou_median = _get_iou_median(METRICS_WINDOW_MINUTES)

    existing = _load_state()
    state = {
        "deployed_at": _now_utc_iso(),
        "current_model_s3_key": current_model_s3_key,
        "previous_model_s3_key": previous_model_s3_key,
        "baseline_iou_median": baseline_iou_median,
        "baseline_edit_pixels_p75": baseline_edit_pixels_p75,
        "rollback_count": existing.get("rollback_count", 0),
    }
    _save_state(state)
    print(f"[rollback_monitor] Deploy recorded: {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor serving metrics and roll back if thresholds are breached")
    sub = parser.add_subparsers(dest="cmd")

    check_cmd = sub.add_parser("check", help="Run rollback check (default action)")
    check_cmd.add_argument("--dry-run", action="store_true")

    force_cmd = sub.add_parser("force-rollback", help="Immediately roll back to a previous or specific model version")
    force_cmd.add_argument(
        "--to-version",
        type=int,
        default=None,
        help="MLflow model version to roll back to (default: current Production version - 1)",
    )
    force_cmd.add_argument("--reason", default="manual", help="Reason recorded in the state file audit trail")

    record_cmd = sub.add_parser("record-deploy", help="Record a new deployment so the warmup clock resets")
    record_cmd.add_argument("--current-key",  required=True, help="S3 key of the newly deployed model")
    record_cmd.add_argument("--previous-key", required=True, help="S3 key of the previous model (for rollback)")
    record_cmd.add_argument("--baseline-iou",        type=float, default=None)
    record_cmd.add_argument("--baseline-edit-pixels", type=float, default=None)

    args = parser.parse_args()

    if args.cmd == "force-rollback":
        result = force_rollback(to_version=args.to_version, reason=args.reason)
        print(json.dumps(result, indent=2))
    elif args.cmd == "record-deploy":
        record_deploy(
            current_model_s3_key=args.current_key,
            previous_model_s3_key=args.previous_key,
            baseline_iou_median=args.baseline_iou,
            baseline_edit_pixels_p75=args.baseline_edit_pixels,
        )
    else:
        dry_run = getattr(args, "dry_run", False)
        result = check_and_rollback(dry_run=dry_run)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

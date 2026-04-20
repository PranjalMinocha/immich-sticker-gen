import csv
import io
import json
import os
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, List, Tuple

import boto3


S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")


@dataclass
class CompiledRetrainingDataset:
    retrain_run_id: str
    total_rows: int
    train_count: int
    val_count: int
    skipped_count: int
    train_manifest_s3_uri: str
    val_manifest_s3_uri: str
    metadata_s3_uri: str
    prefix: str
    accepted_generation_ids: List[str]


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


def _parse_json_text(raw: Any, field_name: str):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be json text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field_name}") from exc


def _sanitize_bbox(raw_bbox: Any) -> List[float]:
    bbox = _parse_json_text(raw_bbox, "bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    values = [float(v) for v in bbox]
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("bbox width/height must be positive")
    return values


def _sanitize_mask_rle(raw_mask: Any) -> Dict[str, Any]:
    payload = _parse_json_text(raw_mask, "userSavedMask")
    if not isinstance(payload, dict):
        raise ValueError("userSavedMask must be an object with size/counts")
    size = payload.get("size")
    counts = payload.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("userSavedMask.size must be [height, width]")
    if not isinstance(counts, list):
        raise ValueError("userSavedMask.counts must be a list")
    height = int(size[0])
    width = int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("userSavedMask size must be positive")

    normalized_counts: List[int] = []
    total = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError("userSavedMask.counts cannot contain negative values")
        total += count
        normalized_counts.append(count)

    if total != height * width:
        raise ValueError("userSavedMask.counts total does not match mask size")

    return {"size": [height, width], "counts": normalized_counts}


def _extension_for_object_key(key: str) -> str:
    lower = (key or "").lower()
    if lower.endswith(".png"):
        return ".png"
    if lower.endswith(".jpeg"):
        return ".jpeg"
    if lower.endswith(".jpg"):
        return ".jpg"
    return ".jpg"


def _build_annotation_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    bbox = _sanitize_bbox(row.get("bbox"))
    mask_payload = _sanitize_mask_rle(row.get("userSavedMask"))
    point_coords = row.get("pointCoords")
    if point_coords is not None:
        point_coords = _parse_json_text(point_coords, "pointCoords")

    return {
        "annotations": [
            {
                "bbox": bbox,
                "segmentation": mask_payload,
                "point_coords": point_coords,
            }
        ]
    }


def _split_rows_deterministic(rows: List[Dict[str, Any]], val_fraction: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sorted_rows = sorted(rows, key=lambda r: str(r["generationId"]))
    total = len(sorted_rows)
    if total < 2:
        raise ValueError("Need at least 2 rows to build train/val manifests")

    val_count = max(1, int(round(total * val_fraction)))
    val_count = min(val_count, total - 1)
    train_count = total - val_count
    return sorted_rows[:train_count], sorted_rows[train_count:]


def _manifest_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buff = io.StringIO()
    writer = csv.writer(buff)
    writer.writerow(["image_uri", "annotation_uri"])
    for row in rows:
        writer.writerow([row["image_uri"], row["annotation_uri"]])
    return buff.getvalue().encode("utf-8")


def compile_retraining_dataset(rows: List[Dict[str, Any]], retrain_run_id: str) -> CompiledRetrainingDataset:
    if not RAW_BUCKET:
        raise RuntimeError("RAW_BUCKET is required")
    if not rows:
        raise ValueError("No rows provided for retraining dataset compile")

    s3_client = _s3_client()
    prefix = f"retraining_runs/{retrain_run_id}"
    images_prefix = f"{prefix}/images"
    annotations_prefix = f"{prefix}/annotations"
    manifests_prefix = f"{prefix}/dataset_manifests"

    accepted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for row in rows:
        generation_id = str(row.get("generationId", ""))
        original_path = row.get("originalPath")
        if not generation_id or not _object_exists(s3_client, RAW_BUCKET, original_path):
            skipped.append({"generationId": generation_id, "reason": "missing_original_image"})
            continue

        try:
            annotation_payload = _build_annotation_payload(row)
        except Exception as exc:
            skipped.append({"generationId": generation_id, "reason": f"invalid_row:{exc}"})
            continue

        extension = _extension_for_object_key(str(original_path))
        image_key = f"{images_prefix}/{generation_id}{extension}"
        annotation_key = f"{annotations_prefix}/{generation_id}.json"

        s3_client.copy_object(
            Bucket=RAW_BUCKET,
            CopySource={"Bucket": RAW_BUCKET, "Key": original_path},
            Key=image_key,
        )
        s3_client.put_object(
            Bucket=RAW_BUCKET,
            Key=annotation_key,
            Body=json.dumps(annotation_payload, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )

        accepted.append(
            {
                "generationId": generation_id,
                "image_uri": f"s3://{RAW_BUCKET}/{image_key}",
                "annotation_uri": f"s3://{RAW_BUCKET}/{annotation_key}",
            }
        )

    train_rows, val_rows = _split_rows_deterministic(accepted)

    train_manifest_key = f"{manifests_prefix}/train_manifest.csv"
    val_manifest_key = f"{manifests_prefix}/val_manifest.csv"
    s3_client.put_object(Bucket=RAW_BUCKET, Key=train_manifest_key, Body=_manifest_csv_bytes(train_rows), ContentType="text/csv")
    s3_client.put_object(Bucket=RAW_BUCKET, Key=val_manifest_key, Body=_manifest_csv_bytes(val_rows), ContentType="text/csv")

    metadata = {
        "retrainRunId": retrain_run_id,
        "createdAt": datetime_utc_iso(),
        "prefix": prefix,
        "totalRowsRequested": len(rows),
        "acceptedRows": len(accepted),
        "skippedRows": len(skipped),
        "trainCount": len(train_rows),
        "valCount": len(val_rows),
        "acceptedGenerationIds": [row["generationId"] for row in accepted],
        "skipped": skipped,
        "contentHash": sha1(json.dumps(accepted, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    metadata_key = f"{prefix}/metadata.json"
    s3_client.put_object(
        Bucket=RAW_BUCKET,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )

    return CompiledRetrainingDataset(
        retrain_run_id=retrain_run_id,
        total_rows=len(rows),
        train_count=len(train_rows),
        val_count=len(val_rows),
        skipped_count=len(skipped),
        train_manifest_s3_uri=f"s3://{RAW_BUCKET}/{train_manifest_key}",
        val_manifest_s3_uri=f"s3://{RAW_BUCKET}/{val_manifest_key}",
        metadata_s3_uri=f"s3://{RAW_BUCKET}/{metadata_key}",
        prefix=prefix,
        accepted_generation_ids=[row["generationId"] for row in accepted],
    )


def datetime_utc_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

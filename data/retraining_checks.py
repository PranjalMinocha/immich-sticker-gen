import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class RetrainingQualityConfig:
    max_edited_pixels_warn: int
    max_num_tries_warn: int
    max_processing_time_ms_warn: int
    max_hard_fail_rate: float
    max_soft_warn_rate: float
    min_accepted_batch_size: int


@dataclass
class ValidationResult:
    status: str
    hard_fail_reasons: List[str]
    soft_warn_reasons: List[str]
    metrics: Dict[str, Any]
    dedupe_key: Optional[str]


def load_quality_config() -> RetrainingQualityConfig:
    return RetrainingQualityConfig(
        max_edited_pixels_warn=_get_int("MAX_EDITED_PIXELS_WARN", 1500),
        max_num_tries_warn=_get_int("MAX_NUM_TRIES_WARN", 4),
        max_processing_time_ms_warn=_get_int("MAX_PROCESSING_TIME_MS_WARN", 5000),
        max_hard_fail_rate=_get_float("MAX_HARD_FAIL_RATE", 0.20),
        max_soft_warn_rate=_get_float("MAX_SOFT_WARN_RATE", 0.80),
        min_accepted_batch_size=_get_int("MIN_ACCEPTED_BATCH_SIZE", 1),
    )


def _parse_json_list(raw_value: Any) -> Tuple[Optional[list], Optional[str]]:
    if raw_value is None:
        return None, "missing_value"

    if isinstance(raw_value, list):
        return raw_value, None

    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return None, "invalid_json"

        if not isinstance(parsed, list):
            return None, "not_list"
        return parsed, None

    return None, "invalid_type"


def _parse_bbox(raw_bbox: Any) -> Tuple[Optional[List[float]], Optional[str]]:
    parsed, err = _parse_json_list(raw_bbox)
    if err:
        return None, err

    if parsed is None or len(parsed) != 4:
        return None, "invalid_length"

    try:
        bbox = [float(v) for v in parsed]
    except (TypeError, ValueError):
        return None, "non_numeric"

    if bbox[2] <= 0 or bbox[3] <= 0:
        return None, "non_positive_area"

    return bbox, None


def _parse_point_coords(raw_point_coords: Any) -> Tuple[Optional[list], Optional[str]]:
    parsed, err = _parse_json_list(raw_point_coords)
    if err:
        return None, err

    if parsed is None or len(parsed) == 0:
        return None, "empty_points"

    return parsed, None


def _validate_required_fields(row: Dict[str, Any], required_fields: List[str]) -> List[str]:
    reasons = []
    for field in required_fields:
        value = row.get(field)
        if value is None:
            reasons.append("missing_required_field:" + field)
            continue
        if isinstance(value, str) and value.strip() == "":
            reasons.append("missing_required_field:" + field)
    return reasons


def _validate_mask_rle(raw_mask: Any) -> Optional[str]:
    if raw_mask is None:
        return "missing_mask"

    if isinstance(raw_mask, str):
        try:
            payload = json.loads(raw_mask)
        except json.JSONDecodeError:
            return "invalid_json"
    elif isinstance(raw_mask, dict):
        payload = raw_mask
    else:
        return "invalid_type"

    if not isinstance(payload, dict):
        return "invalid_schema"

    size = payload.get("size")
    counts = payload.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        return "invalid_size"
    if not isinstance(counts, list):
        return "invalid_counts"

    try:
        height = int(size[0])
        width = int(size[1])
    except (TypeError, ValueError):
        return "invalid_size"

    if height <= 0 or width <= 0:
        return "invalid_size"

    expected = height * width
    total = 0
    for raw_count in counts:
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            return "invalid_count_value"
        if count < 0:
            return "negative_count"
        total += count

    if total != expected:
        return "invalid_counts_total"

    return None


def validate_row(row: Dict[str, Any], cfg: RetrainingQualityConfig) -> ValidationResult:
    hard_fail_reasons: List[str] = []
    soft_warn_reasons: List[str] = []

    required = ["generationId", "userId", "assetId", "createdAt", "userSavedMask"]
    hard_fail_reasons.extend(_validate_required_fields(row, required))

    bbox, bbox_err = _parse_bbox(row.get("bbox"))
    if bbox_err is not None:
        hard_fail_reasons.append("invalid_bbox:" + bbox_err)

    point_coords, points_err = _parse_point_coords(row.get("pointCoords"))
    if points_err is not None:
        hard_fail_reasons.append("invalid_point_coords:" + points_err)

    mask_err = _validate_mask_rle(row.get("userSavedMask"))
    if mask_err is not None:
        hard_fail_reasons.append("invalid_user_saved_mask:" + mask_err)

    edited_pixels = row.get("editedPixels")
    num_tries = row.get("numTries")
    processing_time_ms = row.get("processingTimeMs")

    if edited_pixels is not None and edited_pixels < 0:
        hard_fail_reasons.append("invalid_numeric_range:edited_pixels_negative")
    if num_tries is not None and num_tries < 1:
        hard_fail_reasons.append("invalid_numeric_range:num_tries_lt_1")
    if processing_time_ms is not None and processing_time_ms < 0:
        hard_fail_reasons.append("invalid_numeric_range:processing_time_negative")

    if isinstance(edited_pixels, int) and edited_pixels > cfg.max_edited_pixels_warn:
        soft_warn_reasons.append("high_edited_pixels")
    if isinstance(num_tries, int) and num_tries > cfg.max_num_tries_warn:
        soft_warn_reasons.append("high_num_tries")
    if isinstance(processing_time_ms, int) and processing_time_ms > cfg.max_processing_time_ms_warn:
        soft_warn_reasons.append("high_processing_time_ms")

    metrics: Dict[str, Any] = {"editedPixels": edited_pixels, "numTries": num_tries, "processingTimeMs": processing_time_ms}

    dedupe_key = None
    if bbox is not None and row.get("userId") is not None and row.get("assetId") is not None:
        dedupe_key = "{uid}:{iid}:{x:.2f}:{y:.2f}:{w:.2f}:{h:.2f}".format(
            uid=row["userId"],
            iid=row["assetId"],
            x=bbox[0],
            y=bbox[1],
            w=bbox[2],
            h=bbox[3],
        )

    if hard_fail_reasons:
        status = "hard_fail"
    elif soft_warn_reasons:
        status = "soft_warn"
    else:
        status = "pass"

    return ValidationResult(
        status=status,
        hard_fail_reasons=sorted(set(hard_fail_reasons)),
        soft_warn_reasons=sorted(set(soft_warn_reasons)),
        metrics=metrics,
        dedupe_key=dedupe_key,
    )


def validate_rows(rows: List[Dict[str, Any]], cfg: RetrainingQualityConfig) -> List[Dict[str, Any]]:
    seen_generation_ids = set()
    seen_training_keys = set()
    validated = []

    for row in rows:
        result = validate_row(row, cfg)
        hard_fail_reasons = list(result.hard_fail_reasons)
        soft_warn_reasons = list(result.soft_warn_reasons)

        generation_id = row.get("generationId")
        if generation_id in seen_generation_ids:
            hard_fail_reasons.append("duplicate_generationId")
        else:
            seen_generation_ids.add(generation_id)

        if result.dedupe_key is not None:
            if result.dedupe_key in seen_training_keys:
                hard_fail_reasons.append("duplicate_training_key")
            else:
                seen_training_keys.add(result.dedupe_key)

        status = "hard_fail" if hard_fail_reasons else result.status
        validated.append(
            {
                "row": row,
                "status": status,
                "hard_fail_reasons": sorted(set(hard_fail_reasons)),
                "soft_warn_reasons": sorted(set(soft_warn_reasons)),
                "metrics": result.metrics,
                "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        )

    return validated


def summarize_validation(validated_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(validated_rows)
    accepted_count = sum(1 for v in validated_rows if v["status"] in ("pass", "soft_warn"))
    hard_fail_count = sum(1 for v in validated_rows if v["status"] == "hard_fail")
    soft_warn_count = sum(1 for v in validated_rows if v["status"] == "soft_warn")

    hard_reason_counts: Dict[str, int] = {}
    soft_reason_counts: Dict[str, int] = {}

    for record in validated_rows:
        for reason in record.get("hard_fail_reasons", []):
            hard_reason_counts[reason] = hard_reason_counts.get(reason, 0) + 1
        for reason in record.get("soft_warn_reasons", []):
            soft_reason_counts[reason] = soft_reason_counts.get(reason, 0) + 1

    top_hard = sorted(hard_reason_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    top_soft = sorted(soft_reason_counts.items(), key=lambda item: (-item[1], item[0]))[:10]

    return {
        "total_candidates": total,
        "accepted_count": accepted_count,
        "hard_fail_count": hard_fail_count,
        "soft_warn_count": soft_warn_count,
        "hard_fail_rate": (float(hard_fail_count) / total) if total else 0.0,
        "soft_warn_rate": (float(soft_warn_count) / total) if total else 0.0,
        "top_hard_fail_reasons": top_hard,
        "top_soft_warn_reasons": top_soft,
    }


def should_block_batch(summary: Dict[str, Any], cfg: RetrainingQualityConfig) -> Tuple[bool, List[str]]:
    reasons = []
    if summary["accepted_count"] < cfg.min_accepted_batch_size:
        reasons.append(
            "accepted_count_below_min:{}<{}".format(
                summary["accepted_count"],
                cfg.min_accepted_batch_size,
            )
        )

    if summary["hard_fail_rate"] > cfg.max_hard_fail_rate:
        reasons.append(
            "hard_fail_rate_exceeded:{:.3f}>{:.3f}".format(
                summary["hard_fail_rate"],
                cfg.max_hard_fail_rate,
            )
        )

    if summary["soft_warn_rate"] > cfg.max_soft_warn_rate:
        reasons.append(
            "soft_warn_rate_exceeded:{:.3f}>{:.3f}".format(
                summary["soft_warn_rate"],
                cfg.max_soft_warn_rate,
            )
        )

    return len(reasons) > 0, reasons

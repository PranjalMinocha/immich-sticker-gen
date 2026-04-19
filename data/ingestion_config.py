import os
from dataclasses import dataclass
from typing import Optional


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
class IngestionConfig:
    storage_backend: str
    local_store_root: str
    hard_fail_rate_max: float
    soft_warn_rate_max: float
    min_width: int
    min_height: int
    max_width: int
    max_height: int
    max_aspect_ratio: float
    min_annotations: int
    min_bbox_area_ratio: float
    max_bbox_area_ratio: float
    tiny_bbox_warn_ratio: float
    huge_bbox_warn_ratio: float
    min_blur_variance_warn: float
    min_brightness_warn: float
    max_brightness_warn: float
    max_near_dup_distance: int
    s3_endpoint: Optional[str]
    s3_access_key: Optional[str]
    s3_secret_key: Optional[str]
    raw_bucket: Optional[str]


def load_config() -> IngestionConfig:
    storage_backend = os.environ.get("STORAGE_BACKEND", "local").lower()

    return IngestionConfig(
        storage_backend=storage_backend,
        local_store_root=os.environ.get("LOCAL_STORE_ROOT", "./local_store"),
        hard_fail_rate_max=_get_float("HARD_FAIL_RATE_MAX", 0.15),
        soft_warn_rate_max=_get_float("SOFT_WARN_RATE_MAX", 0.75),
        min_width=_get_int("MIN_WIDTH", 32),
        min_height=_get_int("MIN_HEIGHT", 32),
        max_width=_get_int("MAX_WIDTH", 10000),
        max_height=_get_int("MAX_HEIGHT", 10000),
        max_aspect_ratio=_get_float("MAX_ASPECT_RATIO", 8.0),
        min_annotations=_get_int("MIN_ANNOTATIONS", 1),
        min_bbox_area_ratio=_get_float("MIN_BBOX_AREA_RATIO", 0.000001),
        max_bbox_area_ratio=_get_float("MAX_BBOX_AREA_RATIO", 0.9995),
        tiny_bbox_warn_ratio=_get_float("TINY_BBOX_WARN_RATIO", 0.00005),
        huge_bbox_warn_ratio=_get_float("HUGE_BBOX_WARN_RATIO", 0.95),
        min_blur_variance_warn=_get_float("MIN_BLUR_VARIANCE_WARN", 15.0),
        min_brightness_warn=_get_float("MIN_BRIGHTNESS_WARN", 0.02),
        max_brightness_warn=_get_float("MAX_BRIGHTNESS_WARN", 0.98),
        max_near_dup_distance=_get_int("MAX_NEAR_DUP_DISTANCE", 3),
        s3_endpoint=os.environ.get("S3_ENDPOINT"),
        s3_access_key=os.environ.get("S3_ACCESS_KEY"),
        s3_secret_key=os.environ.get("S3_SECRET_KEY"),
        raw_bucket=os.environ.get("RAW_BUCKET"),
    )


def validate_storage_config(config: IngestionConfig) -> None:
    if config.storage_backend == "s3":
        missing = [
            name
            for name, value in (
                ("S3_ENDPOINT", config.s3_endpoint),
                ("S3_ACCESS_KEY", config.s3_access_key),
                ("S3_SECRET_KEY", config.s3_secret_key),
                ("RAW_BUCKET", config.raw_bucket),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required S3 config env vars: {', '.join(missing)}")

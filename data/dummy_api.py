import io
import json
import os
import random
import threading
import time
import hashlib
import base64
import uuid
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import psycopg2
import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from psycopg2 import Binary
from psycopg2.errors import UniqueViolation
from psycopg2.extras import RealDictCursor

from live_drift import download_detector_artifact, extract_request_features, load_detector


app = FastAPI()
Instrumentator().instrument(app).expose(app)

# S3 / MinIO Setup
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")
SERVING_PREDICT_URL = os.environ.get("SERVING_PREDICT_URL", "http://fastapi_pytorch_server:8000/predict")
SERVING_TIMEOUT_SECONDS = float(os.environ.get("SERVING_TIMEOUT_SECONDS", "30"))

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "database")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "immich")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

SYNTHETIC_USER_EMAIL_PREFIX = os.environ.get("SYNTHETIC_USER_EMAIL_PREFIX", "synthetic+")

ENABLE_DRIFT_MONITORING = os.environ.get("ENABLE_DRIFT_MONITORING", "true").lower() == "true"
DRIFT_DETECTOR_PATH = os.environ.get("DRIFT_DETECTOR_PATH", "/tmp/drift_detector/cd")
DRIFT_WORKERS = int(os.environ.get("DRIFT_WORKERS", "2"))
DRIFT_ARTIFACT_BUCKET = os.environ.get("DRIFT_ARTIFACT_BUCKET", RAW_BUCKET)
DRIFT_ARTIFACT_KEY = os.environ.get("DRIFT_ARTIFACT_KEY", "drift_detectors/initial_training/cd.tar.gz")
DRIFT_LOCAL_CACHE_DIR = os.environ.get("DRIFT_LOCAL_CACHE_DIR", "/tmp/drift_detector")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


STICKER_GENERATE_REQUESTS = Counter(
    "sticker_generate_requests_total",
    "Total /sticker/generate requests",
)
STICKER_EDIT_REQUESTS = Counter(
    "sticker_edit_requests_total",
    "Total /sticker/edit requests",
)
STICKER_RESOLVE_REQUESTS = Counter(
    "sticker_resolve_requests_total",
    "Total /sticker/resolve requests by outcome",
    ["saved"],
)
STICKER_API_ERRORS = Counter(
    "sticker_api_errors_total",
    "Total sticker API errors by endpoint",
    ["endpoint"],
)

STICKER_PROCESSING_TIME_MS = Histogram(
    "sticker_processing_time_ms",
    "Sticker processing time in milliseconds",
    buckets=(50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000),
)
STICKER_EDITED_PIXELS = Histogram(
    "sticker_edited_pixels",
    "Edited pixels per edit operation",
    buckets=(100, 300, 500, 1000, 2000, 4000, 8000, 12000),
)
STICKER_NUM_TRIES = Histogram(
    "sticker_num_tries",
    "Number of tries before sticker resolution",
    buckets=(1, 2, 3, 4, 5, 7, 10),
)
STICKER_BBOX_AREA = Histogram(
    "sticker_bbox_area",
    "Bounding-box area submitted to sticker generate",
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000),
)
STICKER_POINT_COUNT = Histogram(
    "sticker_point_count",
    "Point count in generate requests",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
)

DRIFT_EVENTS = Counter(
    "drift_events_total",
    "Total drift events detected",
)
DRIFT_TEST_STAT = Histogram(
    "drift_test_stat",
    "Online drift detector test statistic",
    buckets=(0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)
DRIFT_DETECTOR_ERRORS = Counter(
    "drift_detector_errors_total",
    "Total errors from online drift detector",
)
DRIFT_FEATURE_ERRORS = Counter(
    "drift_feature_extraction_errors_total",
    "Total feature extraction failures for drift checks",
)


def _initialize_drift_detector():
    if not ENABLE_DRIFT_MONITORING:
        return None

    detector_dir = DRIFT_DETECTOR_PATH
    try:
        if DRIFT_ARTIFACT_BUCKET and DRIFT_ARTIFACT_KEY:
            detector_dir = download_detector_artifact(
                s3,
                DRIFT_ARTIFACT_BUCKET,
                DRIFT_ARTIFACT_KEY,
                DRIFT_LOCAL_CACHE_DIR,
            )
        detector = load_detector(detector_dir)
        if detector is None:
            print(f"Drift detector unavailable at {detector_dir}; drift monitoring disabled")
        else:
            print(
                "Loaded drift detector from s3://{}/{} (local: {})".format(
                    DRIFT_ARTIFACT_BUCKET,
                    DRIFT_ARTIFACT_KEY,
                    detector_dir,
                )
            )
        return detector
    except Exception as exc:
        print(f"Failed to initialize drift detector from object storage: {exc}")
        return None


drift_detector = _initialize_drift_detector()
drift_executor = ThreadPoolExecutor(max_workers=max(1, DRIFT_WORKERS))
drift_lock = threading.Lock()


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def _ensure_json(value: str, field_name: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON for '{field_name}'") from exc


def _mask_to_rle(mask: np.ndarray) -> str:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array")

    flattened = mask.astype(np.uint8).flatten(order="F")
    counts = []
    current_value = 0
    run_length = 0

    for pixel in flattened:
        if pixel == current_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current_value = int(pixel)

    counts.append(run_length)
    return json.dumps({"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts})


def _rle_to_mask(mask_rle: str) -> np.ndarray:
    try:
        payload = json.loads(mask_rle)
    except json.JSONDecodeError as exc:
        raise ValueError("Mask RLE is not valid JSON") from exc

    if not isinstance(payload, dict) or "size" not in payload or "counts" not in payload:
        raise ValueError("Mask RLE payload must include 'size' and 'counts'")

    size = payload["size"]
    counts = payload["counts"]
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Mask RLE 'size' must be [height, width]")
    if not isinstance(counts, list):
        raise ValueError("Mask RLE 'counts' must be a list")

    height = int(size[0])
    width = int(size[1])
    total = height * width
    if height <= 0 or width <= 0:
        raise ValueError("Mask RLE size must be positive")

    flattened = np.zeros(total, dtype=np.uint8)
    index = 0
    value = 0

    for raw_count in counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError("Mask RLE counts must be non-negative")
        if count == 0:
            value = 1 - value
            continue

        next_index = index + count
        if next_index > total:
            raise ValueError("Mask RLE counts exceed expected mask size")

        if value == 1:
            flattened[index:next_index] = 1
        index = next_index
        value = 1 - value

    if index != total:
        raise ValueError("Mask RLE counts do not fill expected mask size")

    return flattened.reshape((height, width), order="F").astype(bool)


def _sanitize_bbox(bbox_data, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    if not isinstance(bbox_data, list) or len(bbox_data) != 4:
        raise ValueError("bbox must be a list of [x, y, width, height]")

    x, y, width, height = [float(coord) for coord in bbox_data]
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(image_width, int(np.ceil(x + width)))
    y2 = min(image_height, int(np.ceil(y + height)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox collapses to an empty crop")

    return x1, y1, x2, y2


def _render_sticker_png(image_bytes: bytes, mask_rle: str, bbox_data) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    image_array = np.array(image)
    image_height, image_width = image_array.shape[:2]

    mask = _rle_to_mask(mask_rle)
    if mask.shape != (image_height, image_width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {(image_height, image_width)}"
        )

    x1, y1, x2, y2 = _sanitize_bbox(bbox_data, image_width, image_height)

    cropped_rgba = image_array[y1:y2, x1:x2, :].copy()
    cropped_alpha = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    cropped_rgba[:, :, 3] = cropped_alpha

    output = io.BytesIO()
    Image.fromarray(cropped_rgba, mode="RGBA").save(output, format="PNG")
    return output.getvalue()


def _generate_mask_from_serving(image_bytes: bytes, bbox_data, point_coords_data) -> tuple[str, int]:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "image": image_b64,
        "bbox": bbox_data,
        "point_coords": point_coords_data,
    }

    started_at = time.perf_counter()
    response = requests.post(SERVING_PREDICT_URL, json=payload, timeout=SERVING_TIMEOUT_SECONDS)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Serving request failed: {exc}") from exc

    body = response.json()
    if "mask" not in body:
        raise HTTPException(status_code=502, detail="Serving response did not include 'mask'")

    mask_png = base64.b64decode(body["mask"])
    mask_array = np.array(Image.open(io.BytesIO(mask_png)).convert("L")) > 127
    rle = _mask_to_rle(mask_array)

    inference_ms = body.get("inference_ms")
    if inference_ms is None:
        return rle, elapsed_ms
    return rle, int(inference_ms)


def _run_drift_check_async(feature_vector) -> None:
    if drift_detector is None:
        return
    try:
        with drift_lock:
            prediction = drift_detector.predict(feature_vector[0])
        payload = prediction.get("data", {}) if isinstance(prediction, dict) else {}
        test_stat = float(payload.get("test_stat", 0.0))
        is_drift = int(payload.get("is_drift", 0))
        DRIFT_TEST_STAT.observe(test_stat)
        if is_drift:
            DRIFT_EVENTS.inc()
    except Exception:
        DRIFT_DETECTOR_ERRORS.inc()


@app.post("/users/register")
def register_user(username: str = Form(...), email: str = Form(...), ml_opt_in: bool = Form(...)):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            'INSERT INTO "user" ("name", "email", "mlTrainingOptIn") VALUES (%s, %s, %s) RETURNING "id";',
            (username, email, ml_opt_in),
        )
        user_id = cur.fetchone()["id"]
        conn.commit()
        return {"user_id": user_id, "status": "registered"}
    finally:
        cur.close()
        conn.close()


@app.get("/users/synthetic")
def get_synthetic_users(limit: int = Query(default=100, ge=1, le=1000)):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            'SELECT "id", "name", "email", "mlTrainingOptIn" FROM "user" WHERE "email" LIKE %s ORDER BY "createdAt" DESC LIMIT %s;',
            (f"{SYNTHETIC_USER_EMAIL_PREFIX}%", limit),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return {
        "users": [
            {
                "user_id": row["id"],
                "username": row["name"],
                "email": row["email"],
                "mlTrainingOptIn": row["mlTrainingOptIn"],
            }
            for row in rows
        ]
    }


@app.post("/upload")
async def upload_image(user_id: str = Form(...), file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_size = len(file_bytes)
    checksum = hashlib.sha1(file_bytes).digest()

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute('SELECT "id" FROM "user" WHERE "id" = %s LIMIT 1;', (user_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="User not found")

        cur.execute('SELECT "id" FROM "asset" WHERE "ownerId" = %s AND "checksum" = %s LIMIT 1;', (user_id, Binary(checksum)))
        existing = cur.fetchone()
        if existing:
            return {"asset_id": existing["id"], "image_id": existing["id"], "status": "duplicate", "s3_object_key": None}

        device_asset_id = uuid.uuid4().hex
        device_id = "synthetic-generator"
        asset_uuid = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        object_key = f"upload/{user_id}/{asset_uuid[:2]}/{asset_uuid[2:4]}/{asset_uuid}{ext}"

        s3.put_object(Bucket=RAW_BUCKET, Key=object_key, Body=file_bytes)

        cur.execute(
            '''
            INSERT INTO "asset"
            ("id", "deviceAssetId", "deviceId", "ownerId", "type", "originalPath", "fileCreatedAt", "fileModifiedAt", "checksum", "checksumAlgorithm", "originalFileName", "localDateTime", "visibility", "libraryId", "livePhotoVideoId", "isExternal")
            VALUES (%s, %s, %s, %s, %s, %s, now(), now(), %s, %s, %s, now(), %s, NULL, NULL, false)
            RETURNING "id";
            ''',
            (asset_uuid, device_asset_id, device_id, user_id, "IMAGE", object_key, Binary(checksum), "sha1", file.filename, "timeline"),
        )
        asset_id = cur.fetchone()["id"]
    except Exception:
        conn.rollback()
        STICKER_API_ERRORS.labels(endpoint="upload").inc()
        raise

    try:
        cur.execute(
            'INSERT INTO "asset_exif" ("assetId", "fileSizeInByte") VALUES (%s, %s) ON CONFLICT ("assetId") DO NOTHING;',
            (asset_id, file_size),
        )
        cur.execute('INSERT INTO "asset_job_status" ("assetId") VALUES (%s) ON CONFLICT ("assetId") DO NOTHING;', (asset_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        STICKER_API_ERRORS.labels(endpoint="upload").inc()
        raise
    finally:
        cur.close()
        conn.close()

    return {"asset_id": asset_id, "image_id": asset_id, "status": "uploaded", "s3_object_key": object_key}


@app.post("/sticker/generate")
def generate_initial_sticker(
    asset_id: str | None = Form(None),
    image_id: str | None = Form(None),
    user_id: str = Form(...),
    bbox: str = Form(...),
    point_coords: str = Form(...),
):
    STICKER_GENERATE_REQUESTS.inc()

    asset_id = asset_id or image_id
    if not asset_id:
        raise HTTPException(status_code=422, detail="Provide asset_id")

    bbox_data = _ensure_json(bbox, "bbox")
    point_coords_data = _ensure_json(point_coords, "point_coords")

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute('SELECT "originalPath", "ownerId" FROM "asset" WHERE "id" = %s LIMIT 1;', (asset_id,))
        asset_row = cur.fetchone()
        if asset_row is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        if asset_row["ownerId"] != user_id:
            raise HTTPException(status_code=403, detail="Asset does not belong to user")

        img_obj = s3.get_object(Bucket=RAW_BUCKET, Key=asset_row["originalPath"])
        image_bytes = img_obj["Body"].read()
        ml_suggested_mask, processing_ms = _generate_mask_from_serving(image_bytes, bbox_data, point_coords_data)

        cur.execute(
            """
            INSERT INTO sticker_generation
            ("source", "assetId", "userId", "bbox", "pointCoords", "mlSuggestedMask", "processingTimeMs", "saved", "numTries", "editedPixels")
            VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, 1, 0)
            RETURNING "id";
            """,
            ("synthetic", asset_id, user_id, json.dumps(bbox_data), json.dumps(point_coords_data), ml_suggested_mask, processing_ms),
        )
        gen_id = cur.fetchone()["id"]
        conn.commit()
    except Exception:
        conn.rollback()
        STICKER_API_ERRORS.labels(endpoint="sticker_generate").inc()
        raise
    finally:
        cur.close()
        conn.close()

    STICKER_PROCESSING_TIME_MS.observe(processing_ms)

    feature_vector = extract_request_features(bbox, point_coords)
    if feature_vector is None:
        DRIFT_FEATURE_ERRORS.inc()
    else:
        bbox_area = float(feature_vector[0][4])
        point_count = float(feature_vector[0][8])
        STICKER_BBOX_AREA.observe(bbox_area)
        STICKER_POINT_COUNT.observe(point_count)
        if drift_detector is not None:
            drift_executor.submit(_run_drift_check_async, feature_vector)

    return {"generation_id": gen_id, "ml_suggested_mask": ml_suggested_mask}


@app.put("/sticker/edit")
def edit_sticker(generation_id: str = Form(...), new_edited_pixels: int = Form(...)):
    STICKER_EDIT_REQUESTS.inc()
    processing_time = random.uniform(0.2, 0.8)
    time.sleep(processing_time)

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            UPDATE sticker_generation
            SET "numTries" = "numTries" + 1,
                "editedPixels" = "editedPixels" + %s,
                "processingTimeMs" = "processingTimeMs" + %s
            WHERE "id" = %s
            RETURNING "numTries", "editedPixels", "processingTimeMs";
            """,
            (new_edited_pixels, int(processing_time * 1000), generation_id),
        )
        row = cur.fetchone()
        conn.commit()
    except Exception:
        STICKER_API_ERRORS.labels(endpoint="sticker_edit").inc()
        raise
    finally:
        cur.close()
        conn.close()

    STICKER_EDITED_PIXELS.observe(max(0, new_edited_pixels))
    if row:
        STICKER_NUM_TRIES.observe(max(1, int(row["numTries"])))
        STICKER_PROCESSING_TIME_MS.observe(max(0, int(row["processingTimeMs"])))

    return {"status": "edited", "generation_id": generation_id}


@app.post("/sticker/resolve")
def resolve_sticker(
    generation_id: str = Form(...),
    saved: bool = Form(...),
    user_saved_mask: str = Form(...),
):
    STICKER_RESOLVE_REQUESTS.labels(saved=str(saved).lower()).inc()

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    s3_sticker_key = None

    if saved:
        try:
            cur.execute(
                """
                SELECT sg."bbox", sg."userId", sg."mlSuggestedMask", sg."userSavedMask", a."originalPath"
                FROM sticker_generation sg
                JOIN "asset" a ON sg."assetId" = a."id"
                WHERE sg."id" = %s;
                """,
                (generation_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Generation not found")

            img_obj = s3.get_object(Bucket=RAW_BUCKET, Key=row["originalPath"])
            image_bytes = img_obj["Body"].read()

            bbox_data = row["bbox"]
            if isinstance(bbox_data, str):
                bbox_data = json.loads(bbox_data)

            effective_mask = user_saved_mask or row.get("userSavedMask") or row.get("mlSuggestedMask")
            if not effective_mask:
                raise ValueError("No mask available for sticker rendering")

            try:
                png_bytes = _render_sticker_png(image_bytes, effective_mask, bbox_data)
            except ValueError:
                fallback_mask = row.get("mlSuggestedMask")
                if not fallback_mask:
                    raise
                png_bytes = _render_sticker_png(image_bytes, fallback_mask, bbox_data)

            s3_sticker_key = f"stickers/user_{row['userId']}/gen_{generation_id}.png"
            s3.put_object(Bucket=RAW_BUCKET, Key=s3_sticker_key, Body=png_bytes, ContentType="image/png")
            print(f"SUCCESS: Sticker saved to {s3_sticker_key}")

        except Exception as exc:
            STICKER_API_ERRORS.labels(endpoint="sticker_resolve").inc()
            print(f"CRITICAL ERROR processing sticker for gen_id {generation_id}: {exc}")

    cur.execute(
        """
        UPDATE sticker_generation
        SET "saved" = %s, "userSavedMask" = %s, "s3StickerKey" = %s
        WHERE "id" = %s;
        """,
        (saved, user_saved_mask, s3_sticker_key, generation_id),
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"status": "resolved", "saved": saved, "s3_sticker_key": s3_sticker_key}

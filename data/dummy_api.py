import io
import json
import os
import random
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import psycopg2
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from psycopg2.extras import RealDictCursor

from live_drift import download_detector_artifact, extract_request_features, load_detector


app = FastAPI()
Instrumentator().instrument(app).expose(app)

# S3 / MinIO Setup
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

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
    return psycopg2.connect(host="postgres", database="sticker_gen", user="admin", password="password123")


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
            "INSERT INTO users (username, email, ml_training_opt_in) VALUES (%s, %s, %s) RETURNING user_id;",
            (username, email, ml_opt_in),
        )
        user_id = cur.fetchone()["user_id"]
        conn.commit()
        return {"user_id": user_id, "status": "registered"}
    finally:
        cur.close()
        conn.close()


@app.post("/upload")
async def upload_image(user_id: int = Form(...), file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_size = len(file_bytes)
    object_key = f"user_uploads/{user_id}/{file.filename}"

    s3.put_object(Bucket=RAW_BUCKET, Key=object_key, Body=file_bytes)

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "INSERT INTO image_uploads (user_id, s3_object_key, original_filename, file_size_bytes) VALUES (%s, %s, %s, %s) RETURNING image_id;",
        (user_id, object_key, file.filename, file_size),
    )
    image_id = cur.fetchone()["image_id"]
    conn.commit()
    cur.close()
    conn.close()

    return {"image_id": image_id, "status": "uploaded"}


@app.post("/sticker/generate")
def generate_initial_sticker(
    image_id: int = Form(...),
    user_id: int = Form(...),
    bbox: str = Form(...),
    point_coords: str = Form(...),
):
    STICKER_GENERATE_REQUESTS.inc()
    processing_time = random.uniform(0.5, 1.5)
    time.sleep(processing_time)

    mock_rle_mask = "RLE_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=40))

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            INSERT INTO sticker_generations
            (image_id, user_id, bbox, point_coords, ml_suggested_mask, processing_time_ms, saved, num_tries, edited_pixels)
            VALUES (%s, %s, %s, %s, %s, %s, NULL, 1, 0) RETURNING generation_id;
            """,
            (image_id, user_id, bbox, point_coords, mock_rle_mask, int(processing_time * 1000)),
        )
        gen_id = cur.fetchone()["generation_id"]
        conn.commit()
    except Exception:
        STICKER_API_ERRORS.labels(endpoint="sticker_generate").inc()
        raise
    finally:
        cur.close()
        conn.close()

    processing_ms = int(processing_time * 1000)
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

    return {"generation_id": gen_id, "ml_suggested_mask": mock_rle_mask}


@app.put("/sticker/edit")
def edit_sticker(generation_id: int = Form(...), new_edited_pixels: int = Form(...)):
    STICKER_EDIT_REQUESTS.inc()
    processing_time = random.uniform(0.2, 0.8)
    time.sleep(processing_time)

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            UPDATE sticker_generations
            SET num_tries = num_tries + 1,
                edited_pixels = edited_pixels + %s,
                processing_time_ms = processing_time_ms + %s
            WHERE generation_id = %s
            RETURNING num_tries, edited_pixels, processing_time_ms;
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
        STICKER_NUM_TRIES.observe(max(1, int(row["num_tries"])))
        STICKER_PROCESSING_TIME_MS.observe(max(0, int(row["processing_time_ms"])))

    return {"status": "edited", "generation_id": generation_id}


@app.post("/sticker/resolve")
def resolve_sticker(
    generation_id: int = Form(...),
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
                SELECT sg.bbox, sg.user_id, iu.s3_object_key
                FROM sticker_generations sg
                JOIN image_uploads iu ON sg.image_id = iu.image_id
                WHERE sg.generation_id = %s;
                """,
                (generation_id,),
            )
            row = cur.fetchone()

            img_obj = s3.get_object(Bucket=RAW_BUCKET, Key=row["s3_object_key"])
            image_bytes = img_obj["Body"].read()

            bbox_data = row["bbox"]
            if isinstance(bbox_data, str):
                bbox_data = json.loads(bbox_data)

            x, y, width, height = [int(coord) for coord in bbox_data]

            image = Image.open(io.BytesIO(image_bytes))
            cropped_image = image.crop((x, y, x + width, y + height))
            if cropped_image.mode != "RGB":
                cropped_image = cropped_image.convert("RGB")

            img_byte_arr = io.BytesIO()
            cropped_image.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()

            s3_sticker_key = f"stickers/user_{row['user_id']}/gen_{generation_id}.jpg"
            s3.put_object(Bucket=RAW_BUCKET, Key=s3_sticker_key, Body=img_byte_arr)
            print(f"SUCCESS: Sticker saved to {s3_sticker_key}")

        except Exception as exc:
            STICKER_API_ERRORS.labels(endpoint="sticker_resolve").inc()
            print(f"CRITICAL ERROR processing sticker for gen_id {generation_id}: {exc}")

    cur.execute(
        """
        UPDATE sticker_generations
        SET saved = %s, user_saved_mask = %s, s3_sticker_key = %s
        WHERE generation_id = %s;
        """,
        (saved, user_saved_mask, s3_sticker_key, generation_id),
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"status": "resolved", "saved": saved, "s3_sticker_key": s3_sticker_key}

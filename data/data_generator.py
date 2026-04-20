import csv
import json
import os
import random
import time
from io import BytesIO, StringIO

import boto3
import numpy as np
import requests
from PIL import Image

API_BASE_URL = "http://dummy-api:8000"

ACTIVE_USERS = []
_used_uploads: dict[str, set[str]] = {}  # user_id → set of jpg_paths already uploaded

# --- Chameleon S3 Setup ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)


def load_synthetic_users(limit=500):
    print("Loading synthetic users from dummy API...")
    response = requests.get(f"{API_BASE_URL}/users/synthetic", params={"limit": limit}, timeout=30)
    response.raise_for_status()
    users = response.json().get("users", [])
    if not users:
        raise RuntimeError(
            "No synthetic users found. Seed Immich users with emails starting with SYNTHETIC_USER_EMAIL_PREFIX."
        )
    print(f"Loaded {len(users)} synthetic users.")
    return users

def get_image_json_pairs():
    """Reads the prod_manifest.csv from S3 and returns a list of S3 Keys."""
    print("Fetching production manifest from Chameleon S3...")
    
    # Get the CSV file from S3
    csv_obj = s3.get_object(Bucket=RAW_BUCKET, Key="dataset_manifests/prod_manifest.csv")
    csv_text = csv_obj['Body'].read().decode('utf-8')
    
    pairs = []
    # Parse the CSV to get the image and annotation URIs
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        # row['image_uri'] looks like: s3://transformed_bucket/images_1024/sa_123.jpg
        img_key = row['image_uri'].split(f"{RAW_BUCKET}/")[1]
        ann_key = row['annotation_uri'].split(f"{RAW_BUCKET}/")[1]
        pairs.append((img_key, ann_key))
        
    return pairs

def apply_bbox_noise_expand_only(bbox, image_width, image_height):
    """Expands a bbox without shrinking so the original box remains covered."""
    x, y, w, h = [float(v) for v in bbox]
    if random.random() >= 0.30:
        return [x, y, w, h]

    max_left = max(0.0, x)
    max_up = max(0.0, y)
    max_right = max(0.0, image_width - (x + w))
    max_down = max(0.0, image_height - (y + h))

    extra_left = min(max_left, w * random.uniform(0.0, 0.20))
    extra_up = min(max_up, h * random.uniform(0.0, 0.20))
    extra_right = min(max_right, w * random.uniform(0.0, 0.20))
    extra_down = min(max_down, h * random.uniform(0.0, 0.20))

    return [
        x - extra_left,
        y - extra_up,
        w + extra_left + extra_right,
        h + extra_up + extra_down,
    ]


def _load_mask_utils():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for annotation mask decoding") from exc
    return mask_utils


def _decode_annotation_mask(annotation, image_height, image_width):
    seg = annotation.get("segmentation")
    if seg is None:
        raise ValueError("annotation missing segmentation")

    mask_utils = _load_mask_utils()

    if isinstance(seg, list):
        rle = mask_utils.frPyObjects(seg, image_height, image_width)
    elif isinstance(seg, dict):
        counts = seg.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(seg, image_height, image_width)
        else:
            rle = seg
    else:
        raise ValueError("unsupported segmentation format")

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded > 0, axis=2)
    return np.asarray(decoded > 0, dtype=bool)


def _decode_rle_mask(mask_rle):
    payload = json.loads(mask_rle)
    size = payload.get("size")
    counts = payload.get("counts")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("invalid rle size")
    if not isinstance(counts, list):
        raise ValueError("invalid rle counts")

    height = int(size[0])
    width = int(size[1])
    flat_size = height * width
    flat = np.zeros(flat_size, dtype=np.uint8)

    idx = 0
    value = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 0:
            raise ValueError("invalid negative rle count")
        next_idx = idx + count
        if next_idx > flat_size:
            raise ValueError("rle count overflow")
        if value == 1 and count > 0:
            flat[idx:next_idx] = 1
        idx = next_idx
        value = 1 - value

    if idx != flat_size:
        raise ValueError("rle count total mismatch")

    return flat.reshape((height, width), order="F").astype(bool)


def _encode_uncompressed_rle(mask):
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    flattened = mask.astype(np.uint8).flatten(order="F")
    counts = []
    current_value = 0
    run_length = 0

    for pixel in flattened:
        if int(pixel) == current_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current_value = int(pixel)

    counts.append(run_length)
    return json.dumps({"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts})


def _partition_total(total, parts):
    if parts <= 0:
        return []
    if total <= 0:
        return [0] * parts
    probabilities = np.full(parts, 1.0 / parts, dtype=np.float64)
    return np.random.multinomial(int(total), probabilities).tolist()


def _bbox_from_annotation(annotation):
    raw_bbox = annotation.get("bbox", [0, 0, 0, 0])
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError("annotation bbox must be [x, y, w, h]")
    bbox = [float(v) for v in raw_bbox]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError("annotation bbox must have positive size")
    return bbox


def _default_point_coords(annotation, bbox):
    point_coords = annotation.get("point_coords")
    if isinstance(point_coords, list) and len(point_coords) > 0:
        return point_coords
    return [[bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]]

def process_file_pair(jpg_path, json_path):
    global ACTIVE_USERS

    user = random.choice(ACTIVE_USERS)
    username = user.get('username') or user['email']
    user_id = user['user_id']
    print(f"\n[{username}] Synthetic user selected...")
    
    filename = os.path.basename(jpg_path)
    print(f"[{username}] Fetching {filename} from S3...")
    
    img_obj = s3.get_object(Bucket=RAW_BUCKET, Key=jpg_path)
    img_bytes = img_obj['Body'].read()
    with Image.open(BytesIO(img_bytes)) as image:
        image_width, image_height = image.size
    
    files = {'file': (filename, img_bytes, "image/jpeg")}
    upload_res = requests.post(f"{API_BASE_URL}/upload", data={'user_id': user_id}, files=files, timeout=60)
    upload_res.raise_for_status()

    asset_id = upload_res.json()["asset_id"]
    
    ann_obj = s3.get_object(Bucket=RAW_BUCKET, Key=json_path)
    data = json.loads(ann_obj['Body'].read().decode('utf-8'))
        
    if "annotations" in data and len(data["annotations"]) > 0:
        # Simulate a user creating between 1 and 3 stickers from the same image
        num_stickers = random.randint(1, min(3, len(data["annotations"])))
        selected_anns = random.sample(data["annotations"], num_stickers)
        
        for ann in selected_anns:
            try:
                raw_bbox = _bbox_from_annotation(ann)
            except Exception as exc:
                print(f"  -> [{username}] Skipping annotation due to invalid bbox: {exc}")
                continue

            noisy_bbox = apply_bbox_noise_expand_only(raw_bbox, image_width, image_height)
            point_coords = _default_point_coords(ann, noisy_bbox)

            try:
                gt_mask = _decode_annotation_mask(ann, image_height=image_height, image_width=image_width)
            except Exception as exc:
                print(f"  -> [{username}] Skipping annotation due to invalid segmentation: {exc}")
                continue
            
            # PHASE 1: Initial Generation
            print(f"  -> [{username}] Requesting initial sticker...")
            gen_res = requests.post(
                f"{API_BASE_URL}/sticker/generate",
                data={
                    "asset_id": asset_id,
                    "user_id": user_id,
                    "bbox": json.dumps(noisy_bbox),
                    "point_coords": json.dumps(point_coords),
                },
                timeout=120,
            )
            gen_res.raise_for_status()

            generation_payload = gen_res.json()
            if "generation_id" not in generation_payload:
                continue

            gen_id = generation_payload["generation_id"]
            model_suggested_mask = generation_payload["ml_suggested_mask"]

            try:
                model_mask = _decode_rle_mask(model_suggested_mask)
            except Exception as exc:
                print(f"  -> [{username}] Skipping generation {gen_id} due to invalid model mask: {exc}")
                continue

            if model_mask.shape != gt_mask.shape:
                print(
                    f"  -> [{username}] Skipping generation {gen_id} due to shape mismatch: model {model_mask.shape} vs gt {gt_mask.shape}"
                )
                continue

            total_edited_pixels = int(np.count_nonzero(np.logical_xor(gt_mask, model_mask)))
            gt_mask_rle = _encode_uncompressed_rle(gt_mask)
            
            total_tries = random.choices([1, 2, 3, 4, 5], weights=[60, 20, 10, 5, 5])[0]
            
            # PHASE 2: The Edit Loop
            edit_steps = max(0, total_tries - 1)
            edit_pixels_sequence = _partition_total(total_edited_pixels, edit_steps)

            for attempt_number, pixels_this_edit in enumerate(edit_pixels_sequence, start=2):
                print(f"    -> [{username}] Correcting mask (Attempt {attempt_number})...")

                edit_res = requests.put(
                    f"{API_BASE_URL}/sticker/edit",
                    data={"generation_id": gen_id, "new_edited_pixels": int(pixels_this_edit)},
                    timeout=60,
                )
                edit_res.raise_for_status()
                time.sleep(random.uniform(0.5, 2.0))

            # PHASE 3: Save or Discard
            save_probability = 0.85 if total_tries < 4 else 0.30
            saved = random.random() < save_probability

            action_str = "SAVED" if saved else "DISCARDED"
            print(f"    -> [{username}] {action_str} sticker.")

            # Resolve with the ground-truth annotation mask so userSavedMask reflects true labels.
            resolve_res = requests.post(
                f"{API_BASE_URL}/sticker/resolve",
                data={
                    "generation_id": gen_id,
                    "saved": "true" if saved else "false",
                    "user_saved_mask": gt_mask_rle,
                },
                timeout=60,
            )
            resolve_res.raise_for_status()

if __name__ == "__main__":
    print("Starting Deep-Data Synthetic Generator...")
    time.sleep(5) # Wait for DB and API to boot
    ACTIVE_USERS = load_synthetic_users()

    pairs = get_image_json_pairs()
    print(f"Found {len(pairs)} image/annotation pairs in the reservoir.")

    # Loop through the files indefinitely to keep generating traffic
    while True:
        user = random.choice(ACTIVE_USERS)
        user_id = user["user_id"]
        used = _used_uploads.setdefault(user_id, set())

        # Find a pair this user hasn't uploaded yet.
        candidates = [(j, a) for j, a in pairs if j not in used]
        if not candidates:
            # User has exhausted the whole reservoir; reset their history.
            used.clear()
            candidates = list(pairs)

        jpg_path, json_path = random.choice(candidates)
        used.add(jpg_path)
        process_file_pair(jpg_path, json_path)
        time.sleep(random.uniform(1.0, 4.0))

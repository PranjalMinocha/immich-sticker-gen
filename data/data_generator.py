import requests
import time
import random
import os
import json
import boto3
import csv
from io import StringIO
from faker import Faker

fake = Faker()
API_BASE_URL = "http://dummy-api:8000"
RESERVOIR_DIR = "/app/local_reservoir"

# --- Chameleon S3 Setup ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
TRANSFORMED_BUCKET = os.environ.get("TRANSFORMED_BUCKET")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def get_production_pairs_from_s3():
    """Reads the prod_manifest.csv from S3 and returns a list of S3 Keys."""
    print("Fetching production manifest from Chameleon S3...")
    
    # Get the CSV file from S3
    csv_obj = s3.get_object(Bucket=TRANSFORMED_BUCKET, Key="dataset_manifests/prod_manifest.csv")
    csv_text = csv_obj['Body'].read().decode('utf-8')
    
    pairs = []
    # Parse the CSV to get the image and annotation URIs
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        # row['image_uri'] looks like: s3://transformed_bucket/images_1024/sa_123.jpg
        img_key = row['image_uri'].split(f"{TRANSFORMED_BUCKET}/")[1]
        ann_key = row['annotation_uri'].split(f"{RAW_BUCKET}/")[1]
        pairs.append((img_key, ann_key))
        
    return pairs

def apply_bbox_noise(bbox):
    """Simulates sloppy user drawing by warping the box 30% of the time."""
    if random.random() < 0.30:
        x, y, w, h = bbox
        
        # Scale width and height by +/- 15%
        new_w = w * random.uniform(0.85, 1.15)
        new_h = h * random.uniform(0.85, 1.15)
        
        # Shift the starting x, y coordinates slightly
        new_x = max(0, x + (w * random.uniform(-0.1, 0.1)))
        new_y = max(0, y + (h * random.uniform(-0.1, 0.1)))
        
        return [new_x, new_y, new_w, new_h]
    return bbox

def process_file_pair(jpg_path, json_path):
    username = fake.user_name() + str(random.randint(100, 9999))
    ml_opt_in = random.choice(["true", "false"])
    
    res = requests.post(f"{API_BASE_URL}/users/register", data={
        "username": username, "email": fake.email(), "ml_opt_in": ml_opt_in
    })
    
    if "user_id" not in res.json():
        return
    user_id = res.json()["user_id"]
    
    filename = os.path.basename(jpg_path)
    
    with open(jpg_path, "rb") as f:
        files = {'file': (filename, f, "image/jpeg")}
        upload_res = requests.post(f"{API_BASE_URL}/upload", data={'user_id': user_id}, files=files)
        
    upload_id = upload_res.json()["upload_id"]
    
    with open(json_path, "r") as jf:
        data = json.load(jf)
        
    if "annotations" in data and len(data["annotations"]) > 0:
        # Simulate a user creating between 1 and 3 stickers from the same image
        num_stickers = random.randint(1, min(3, len(data["annotations"])))
        selected_anns = random.sample(data["annotations"], num_stickers)
        
        for ann in selected_anns:
            raw_bbox = ann.get("bbox", [0, 0, 0, 0])
            noisy_bbox = apply_bbox_noise(raw_bbox) # Apply our 30% noise chance
            
            point_coords = ann.get("point_coords", [[noisy_bbox[0] + noisy_bbox[2]/2, noisy_bbox[1] + noisy_bbox[3]/2]])
            
            num_tries = random.choices([1, 2, 3, 4, 5], weights=[60, 20, 10, 5, 5])[0]
            edited_pixels = random.randint(500, 15000) if num_tries > 1 else 0
            save_probability = 0.85 if num_tries < 4 else 0.30
            saved = random.random() < save_probability

            print(f"  -> [{username}] Generating sticker (tries: {num_tries}, saved: {saved})...")
            requests.post(f"{API_BASE_URL}/generate_sticker", data={
                "upload_id": upload_id,
                "bbox": json.dumps(noisy_bbox),
                "point_coords": json.dumps(point_coords),
                "saved": saved,
                "num_tries": num_tries,
                "edited_pixels": edited_pixels
            })

if __name__ == "__main__":
    print("Starting Deep-Data Synthetic Generator...")
    time.sleep(5) # Wait for DB and API to boot
    
    pairs = get_image_json_pairs()
    print(f"Found {len(pairs)} image/annotation pairs in the reservoir.")
    
    # Loop through the files indefinitely to keep generating traffic
    while True:
        random.shuffle(pairs) # Shuffle so traffic looks organic
        for jpg_path, json_path in pairs:
            process_file_pair(jpg_path, json_path)
            time.sleep(random.uniform(1.0, 4.0)) # Wait before next user
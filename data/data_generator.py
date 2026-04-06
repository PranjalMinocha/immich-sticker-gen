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

ACTIVE_USERS = []

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
    global ACTIVE_USERS
    
    if ACTIVE_USERS and random.random() < 0.70:
        user = random.choice(ACTIVE_USERS)
        username = user['username']
        user_id = user['user_id']
        print(f"\n[{username}] Returning user logged in...")
    else:
        # Register a brand new user
        username = fake.user_name() + str(random.randint(100, 9999))
        ml_opt_in = random.choice(["true", "false"])
        
        res = requests.post(f"{API_BASE_URL}/users/register", data={
            "username": username, "email": fake.email(), "ml_opt_in": ml_opt_in
        })
        
        if "user_id" not in res.json():
            return
            
        user_id = res.json()["user_id"]
        
        # Save them to the pool so they can return later!
        ACTIVE_USERS.append({"user_id": user_id, "username": username})
        print(f"\n[{username}] Brand new user registered!")
    
    filename = os.path.basename(jpg_path)
    print(f"[{username}] Fetching {filename} from S3...")
    
    img_obj = s3.get_object(Bucket=RAW_BUCKET, Key=jpg_path)
    img_bytes = img_obj['Body'].read()
    
    files = {'file': (filename, img_bytes, "image/jpeg")}
    upload_res = requests.post(f"{API_BASE_URL}/upload", data={'user_id': user_id}, files=files)
    
    image_id = upload_res.json()["image_id"]
    
    ann_obj = s3.get_object(Bucket=RAW_BUCKET, Key=json_path)
    data = json.loads(ann_obj['Body'].read().decode('utf-8'))
        
    if "annotations" in data and len(data["annotations"]) > 0:
        # Simulate a user creating between 1 and 3 stickers from the same image
        num_stickers = random.randint(1, min(3, len(data["annotations"])))
        selected_anns = random.sample(data["annotations"], num_stickers)
        
        for ann in selected_anns:
            raw_bbox = ann.get("bbox", [0, 0, 0, 0])
            noisy_bbox = apply_bbox_noise(raw_bbox) # Apply our 30% noise chance
            
            point_coords = ann.get("point_coords", [[noisy_bbox[0] + noisy_bbox[2]/2, noisy_bbox[1] + noisy_bbox[3]/2]])
            
            # PHASE 1: Initial Generation
            print(f"  -> [{username}] Requesting initial sticker...")
            gen_res = requests.post(f"{API_BASE_URL}/sticker/generate", data={
                "image_id": image_id,
                "user_id": user_id,
                "bbox": json.dumps(noisy_bbox),
                "point_coords": json.dumps(point_coords)
            })
            
            if "generation_id" not in gen_res.json():
                continue
                
            gen_id = gen_res.json()["generation_id"]
            current_mask_state = gen_res.json()["ml_suggested_mask"] # Capture the model's guess
            
            total_tries = random.choices([1, 2, 3, 4, 5], weights=[60, 20, 10, 5, 5])[0]
            
            # PHASE 2: The Edit Loop
            for attempt in range(total_tries - 1):
                pixels_this_edit = random.randint(500, 5000)
                print(f"    -> [{username}] Correcting mask (Attempt {attempt + 2})...")
                
                # Simulate the user editing the mask by altering the RLE string
                current_mask_state += f"_edited_v{attempt + 1}"
                
                requests.put(f"{API_BASE_URL}/sticker/edit", data={
                    "generation_id": gen_id,
                    "new_edited_pixels": pixels_this_edit
                })
                time.sleep(random.uniform(0.5, 2.0)) 
            
            # PHASE 3: Save or Discard
            save_probability = 0.85 if total_tries < 4 else 0.30
            saved = random.random() < save_probability
            
            action_str = "SAVED" if saved else "DISCARDED"
            print(f"    -> [{username}] {action_str} sticker.")
            
            # Send the final state of the mask back to the database
            requests.post(f"{API_BASE_URL}/sticker/resolve", data={
                "generation_id": gen_id,
                "saved": "true" if saved else "false",
                "user_saved_mask": current_mask_state
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
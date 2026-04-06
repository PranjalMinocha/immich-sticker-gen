import os
import boto3
import csv
import argparse
from sklearn.model_selection import train_test_split

# --- Configuration (Loaded automatically from .env) ---
S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
RAW_BUCKET = os.environ["RAW_BUCKET"]

s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def get_bucket_keys(bucket, prefix):
    """Uses a paginator to get all files in a bucket prefix (handles >1000 files)."""
    paginator = s3.get_paginator('list_objects_v2')
    keys = []
    print(f"Scanning s3://{bucket}/{prefix}...")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                keys.append(obj['Key'])
    return keys

def create_manifest(filename, base_names):
    """Generates a CSV mapping the image URIs to the annotation URIs."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_uri', 'annotation_uri'])
        
        for base in base_names:
            img_uri = f"s3://{RAW_BUCKET}/images/{base}.jpg"
            ann_uri = f"s3://{RAW_BUCKET}/annotations/{base}.json"
            writer.writerow([img_uri, ann_uri])

def main(train_pct=0.7, val_pct=0.15, prod_pct=0.15, seed=42):
    # 1. Fetch all keys from the buckets
    image_keys = get_bucket_keys(RAW_BUCKET, 'images/')
    annotation_keys = get_bucket_keys(RAW_BUCKET, 'annotations/')
    
    # 2. Extract base names (e.g., 'sa_12345' from 'images_1024/sa_12345.jpg')
    image_bases = set([os.path.splitext(os.path.basename(k))[0] for k in image_keys if k.endswith('.jpg')])
    annotation_bases = set([os.path.splitext(os.path.basename(k))[0] for k in annotation_keys if k.endswith('.json')])
    
    # 3. Find valid pairs (images that have a matching annotation)
    valid_bases = list(image_bases.intersection(annotation_bases))
    print(f"Found {len(valid_bases)} valid image/mask pairs in the cloud.")
    
    if not valid_bases:
        print("Error: No data pairs found. Did the ingestion script finish running?")
        return

    # 4. Perform the split
    train_bases, temp_bases = train_test_split(valid_bases, train_size=train_pct, random_state=seed)
    
    relative_val_pct = val_pct / (val_pct + prod_pct)
    val_bases, prod_bases = train_test_split(temp_bases, train_size=relative_val_pct, random_state=seed)

    print(f"Split results -> Train: {len(train_bases)} | Val: {len(val_bases)} | Prod: {len(prod_bases)}")

    # 5. Generate Manifest CSVs locally
    manifests = {
        'train_manifest.csv': train_bases,
        'val_manifest.csv': val_bases,
        'prod_manifest.csv': prod_bases
    }
    
    print("Generating and uploading manifest files...")
    for filename, bases in manifests.items():
        create_manifest(filename, bases)
        
        # Upload the manifest back to the Transformed Bucket for safekeeping
        s3.upload_file(filename, RAW_BUCKET, f"dataset_manifests/{filename}")
        print(f"Uploaded: s3://{RAW_BUCKET}/dataset_manifests/{filename}")
        
        # Clean up local CSV file
        os.remove(filename)
        
    print("Cloud-native split completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(seed=args.seed)
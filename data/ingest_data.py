import os
import tarfile
import requests
import boto3
import argparse
from tqdm import tqdm

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "admin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "password123")

RAW_BUCKET = "objstore-proj28"

s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def download_file(url, local_filename):
    """Downloads a file from a URL with a progress bar."""
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    return local_filename

def process_and_upload(extract_dir):
    """Iterates through extracted files, uploads them, and deletes local copies to save disk space."""
    print("Uploading to Chameleon Cloud Object Storage...")
    
    files = [f for f in os.listdir(extract_dir) if os.path.isfile(os.path.join(extract_dir, f))]
    
    for filename in tqdm(files, desc="Uploading and cleaning up"):
        filepath = os.path.join(extract_dir, filename)
        
        try:
            if filename.endswith('.json'):
                # Upload raw mask annotations directly
                s3.upload_file(filepath, RAW_BUCKET, f"annotations/{filename}")
                
            elif filename.endswith('.jpg'):
                # Upload Raw Image
                s3.upload_file(filepath, RAW_BUCKET, f"images/{filename}")
                
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
        finally:
            # CRITICAL: Delete the file immediately after uploading to prevent cloud instance out-of-disk errors
            os.remove(filepath)

def main(dataset_url):
    archive_name = "dataset_chunk.tar"
    extract_dir = "./extracted_data"
    
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        download_file(dataset_url, archive_name)
        
        print("Extracting archive...")
        with tarfile.open(archive_name, "r:*") as tar:
            tar.extractall(path=extract_dir)
            
        os.remove(archive_name)
        
        process_and_upload(extract_dir)
        
        print("Ingestion complete! Local storage cleared.")
        
    finally:
        # Failsafe cleanup
        if os.path.exists(archive_name):
            os.remove(archive_name)
        if os.path.exists(extract_dir):
            for f in os.listdir(extract_dir):
                os.remove(os.path.join(extract_dir, f))
            os.rmdir(extract_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SA-1B dataset from a URL to Chameleon Object Storage")
    parser.add_argument("--url", required=True, help="Direct download link to the dataset tar file")
    args = parser.parse_args()
    
    main(args.url)
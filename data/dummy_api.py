from fastapi import FastAPI, UploadFile, File, Form
import psycopg2
from psycopg2.extras import RealDictCursor
import time
import random
import os
import boto3
import json

app = FastAPI()

# S3 / MinIO Setup
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET", "immich-raw-data")

s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

def get_db_connection():
    return psycopg2.connect(
        host="postgres", database="sticker_metrics", user="admin", password="password123"
    )

@app.post("/users/register")
def register_user(username: str = Form(...), email: str = Form(...), ml_opt_in: bool = Form(...)):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "INSERT INTO users (username, email, ml_training_opt_in) VALUES (%s, %s, %s) RETURNING user_id;",
            (username, email, ml_opt_in)
        )
        user_id = cur.fetchone()['user_id']
        conn.commit()
        return {"user_id": user_id, "status": "registered"}
    finally:
        cur.close()
        conn.close()

@app.post("/upload")
async def upload_image(user_id: int = Form(...), file: UploadFile = File(...)):
    # 1. Read file bytes
    file_bytes = await file.read()
    file_size = len(file_bytes)
    object_key = f"user_uploads/{user_id}/{file.filename}"
    
    # 2. Upload to MinIO/S3
    s3.put_object(Bucket=RAW_BUCKET, Key=object_key, Body=file_bytes)
    
    # 3. Log to Database
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "INSERT INTO image_uploads (user_id, s3_object_key, original_filename, file_size_bytes) VALUES (%s, %s, %s, %s) RETURNING upload_id;",
        (user_id, object_key, file.filename, file_size)
    )
    upload_id = cur.fetchone()['upload_id']
    conn.commit()
    cur.close()
    conn.close()
    
    return {"upload_id": upload_id, "status": "uploaded"}

@app.post("/generate_sticker")
def generate_sticker(
    upload_id: int = Form(...), 
    bbox: str = Form(...), 
    point_coords: str = Form(...),
    saved: bool = Form(...),
    num_tries: int = Form(...),
    edited_pixels: int = Form(...)
):
    # Scale processing time based on how many tries it took
    processing_time = random.uniform(0.5, 1.5) * num_tries
    time.sleep(processing_time)
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sticker_generations 
        (upload_id, bbox, point_coords, processing_time_ms, saved, num_tries, edited_pixels) 
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """,
        (upload_id, bbox, point_coords, int(processing_time * 1000), saved, num_tries, edited_pixels)
    )
    conn.commit()
    cur.close()
    conn.close()
    
    return {"upload_id": upload_id, "saved": saved, "tries": num_tries}
from fastapi import FastAPI, UploadFile, File, Form
import psycopg2
from psycopg2.extras import RealDictCursor
import time
import random
import os
import boto3
import json
import string
from PIL import Image
import io

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
        host="postgres", database="sticker_gen", user="admin", password="password123"
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
        "INSERT INTO image_uploads (user_id, s3_object_key, original_filename, file_size_bytes) VALUES (%s, %s, %s, %s) RETURNING image_id;",
        (user_id, object_key, file.filename, file_size)
    )
    image_id = cur.fetchone()['image_id']
    conn.commit()
    cur.close()
    conn.close()
    
    return {"image_id": image_id, "status": "uploaded"}

@app.post("/sticker/generate")
def generate_initial_sticker(image_id: int = Form(...), user_id: int = Form(...), bbox: str = Form(...), point_coords: str = Form(...)):
    """Phase 1: User draws the box, model makes its first guess."""
    processing_time = random.uniform(0.5, 1.5)
    time.sleep(processing_time)
    
    # Simulate an ML model outputting an RLE compressed mask string
    mock_rle_mask = "RLE_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=40))
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        INSERT INTO sticker_generations 
        (image_id, user_id, bbox, point_coords, ml_suggested_mask, processing_time_ms, saved, num_tries, edited_pixels) 
        VALUES (%s, %s, %s, %s, %s, %s, NULL, 1, 0) RETURNING generation_id;
        """,
        (image_id, user_id, bbox, point_coords, mock_rle_mask, int(processing_time * 1000)) 
    )
    gen_id = cur.fetchone()['generation_id']
    conn.commit()
    cur.close()
    conn.close()
    
    # Return the mask to the frontend
    return {"generation_id": gen_id, "ml_suggested_mask": mock_rle_mask}

@app.put("/sticker/edit")
def edit_sticker(generation_id: int = Form(...), new_edited_pixels: int = Form(...)):
    """Phase 2: User corrects the mask. This can be called multiple times."""
    processing_time = random.uniform(0.2, 0.8) # Edits are usually faster than initial generation
    time.sleep(processing_time)
    
    conn = get_db_connection()
    cur = conn.cursor()
    # Increment the tries and add the new edited pixels to the running total
    cur.execute(
        """
        UPDATE sticker_generations 
        SET num_tries = num_tries + 1,
            edited_pixels = edited_pixels + %s,
            processing_time_ms = processing_time_ms + %s
        WHERE generation_id = %s;
        """,
        (new_edited_pixels, int(processing_time * 1000), generation_id)
    )
    conn.commit()
    cur.close()
    conn.close()
    
    return {"status": "edited", "generation_id": generation_id}

@app.post("/sticker/resolve")
def resolve_sticker(
    generation_id: int = Form(...), 
    saved: bool = Form(...),
    user_saved_mask: str = Form(...)
):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    s3_sticker_key = None
    
    # Only crop and upload to S3 if the user actually saved the sticker!
    if saved:
        # 1. Fetch the original image location and bounding box
        cur.execute("""
            SELECT sg.bbox, sg.user_id, iu.s3_object_key 
            FROM sticker_generations sg
            JOIN image_uploads iu ON sg.image_id = iu.image_id
            WHERE sg.generation_id = %s;
        """, (generation_id,))
        row = cur.fetchone()
        
        # 2. Download the original image bytes from S3
        img_obj = s3.get_object(Bucket=RAW_BUCKET, Key=row['s3_object_key'])
        image_bytes = img_obj['Body'].read()
        
        # 3. Crop the image using Pillow
        bbox = row['bbox']
        x, y, w, h = [int(coord) for coord in bbox]
        
        image = Image.open(io.BytesIO(image_bytes))
        # Pillow expects a box tuple: (left, upper, right, lower)
        cropped_image = image.crop((x, y, x + w, y + h))
        
        # 4. Save the cropped sticker to memory as a JPEG
        img_byte_arr = io.BytesIO()
        cropped_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 5. Upload the final sticker to S3
        s3_sticker_key = f"stickers/user_{row['user_id']}/gen_{generation_id}.jpg"
        s3.put_object(Bucket=RAW_BUCKET, Key=s3_sticker_key, Body=img_byte_arr)
        
        print(f"Sticker saved physically to S3: {s3_sticker_key}")

    # 6. Update the database with the final status and S3 key
    cur.execute(
        """
        UPDATE sticker_generations 
        SET saved = %s, user_saved_mask = %s, s3_sticker_key = %s 
        WHERE generation_id = %s;
        """, 
        (saved, user_saved_mask, s3_sticker_key, generation_id)
    )
    conn.commit()
    cur.close()
    conn.close()
    
    return {"status": "resolved", "saved": saved, "s3_sticker_key": s3_sticker_key}
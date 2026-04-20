CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) NOT NULL,
    ml_training_opt_in BOOLEAN DEFAULT FALSE, -- New Opt-In Column
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE image_uploads (
    image_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    s3_object_key VARCHAR(255), -- Track where the API saved it in S3/MinIO
    original_filename VARCHAR(255),
    file_size_bytes INTEGER,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sticker_generations (
    generation_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    image_id INTEGER REFERENCES image_uploads(image_id),
    bbox JSONB,
    point_coords JSONB,
    ml_suggested_mask TEXT,
    user_saved_mask TEXT,
    s3_sticker_key VARCHAR(255),
    processing_time_ms INTEGER,
    saved BOOLEAN DEFAULT FALSE,
    num_tries INTEGER DEFAULT 1,
    edited_pixels INTEGER DEFAULT 0,
    quality_status VARCHAR(16) DEFAULT 'pending',
    quality_checked_at TIMESTAMP,
    quality_check_version INTEGER DEFAULT 1,
    quality_fail_reasons_json TEXT,
    used_for_training BOOLEAN DEFAULT FALSE,
    used_for_training_at TIMESTAMP,
    retrain_run_id VARCHAR(64),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

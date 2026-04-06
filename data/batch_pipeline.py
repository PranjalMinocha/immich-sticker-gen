import os
import psycopg2
from psycopg2.extras import execute_values
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

# --- Configuration ---
POSTGRES_URI = os.environ.get("POSTGRES_URI")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

# --- 1. Initialize PySpark with Apache Iceberg ---
spark = SparkSession.builder \
    .appName("ML_Training_Data_Compiler") \
    .config("spark.jars.packages", 
            "org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.3.1,"
            "org.postgresql:postgresql:42.6.0") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.lakehouse", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.lakehouse.type", "hadoop") \
    .config("spark.sql.catalog.lakehouse.warehouse", f"s3a://{RAW_BUCKET}/iceberg_warehouse") \
    .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT) \
    .config("spark.hadoop.fs.s3a.access.key", S3_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", S3_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

def compile_training_batch():
    print("Starting ML Batch Pipeline...")

    # --- 2. Extract Candidate Pool from Postgres ---
    # We push down the basic filters directly to the database for efficiency
    query = """
    (SELECT sg.generation_id, sg.image_id, sg.user_id, sg.s3_sticker_key, 
            sg.ml_suggested_mask, sg.user_saved_mask, iu.s3_object_key as raw_image_key
     FROM sticker_generations sg
     JOIN image_uploads iu ON sg.image_id = iu.image_id
     JOIN users u ON sg.user_id = u.user_id
     WHERE u.ml_training_opt_in = TRUE
       AND sg.saved = TRUE
       AND sg.edited_pixels < 2000
       AND sg.used_for_training = FALSE) AS candidates
    """

    df_candidates = spark.read.format("jdbc") \
        .option("url", POSTGRES_URI) \
        .option("driver", "org.postgresql.Driver") \
        .option("dbtable", query) \
        .option("user", DB_USER) \
        .option("password", DB_PASS) \
        .load()

    # --- 3. Transform: Apply Complex Selection Logic ---
    # Enforce: No more than 10 samples of a single user
    window_spec = Window.partitionBy("user_id").orderBy("generation_id")
    
    df_filtered = df_candidates.withColumn("user_sample_num", row_number().over(window_spec)) \
        .filter(col("user_sample_num") <= 10) \
        .drop("user_sample_num")

    # Enforce: A total of exactly 50 image/sticker pairs
    # (If we have less than 50, we take what we have)
    df_final_batch = df_filtered.limit(50)
    
    batch_count = df_final_batch.count()
    if batch_count == 0:
        print("No new candidate data available for training.")
        return

    print(f"Compiled batch of {batch_count} records. Writing to Iceberg...")

    # --- 4. Load: Write to Iceberg Data Lakehouse ---
    # This automatically versions the data. We use append to add to the existing table.
    df_final_batch.writeTo("lakehouse.ml_datasets.training_data") \
        .tableProperty("write.format.default", "parquet") \
        .append()

    # --- 5. State Update: Mark as Used in Postgres ---
    # Extract the IDs that were just selected to update the source database
    selected_ids = [row.generation_id for row in df_final_batch.select("generation_id").collect()]
    
    conn = psycopg2.connect(host="postgres", database="sticker_metrics", user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    
    # Efficiently update all selected rows in one transaction
    execute_values(
        cur,
        "UPDATE sticker_generations SET used_for_training = TRUE WHERE generation_id IN %s",
        [(id,) for id in selected_ids]
    )
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"Pipeline complete. Successfully versioned {batch_count} rows in Iceberg.")

if __name__ == "__main__":
    compile_training_batch()
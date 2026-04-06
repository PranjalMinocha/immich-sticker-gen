import os
import psycopg2
from psycopg2.extras import execute_values
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

# --- Configuration ---
POSTGRES_URI = os.environ.get("POSTGRES_URI")
POSTGRES_USER = os.environ.get("DB_USER")
POSTGRES_PASSWORD = os.environ.get("DB_PASS")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RAW_BUCKET = os.environ.get("RAW_BUCKET")

# --- 1. Initialize PySpark with Apache Iceberg ---
spark = SparkSession.builder \
    .appName("ML_Training_Data_Compiler") \
    .config("spark.jars.packages", 
            "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0,"
            "org.postgresql:postgresql:42.6.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.lakehouse", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.lakehouse.type", "hadoop") \
    .config("spark.sql.catalog.lakehouse.warehouse", f"s3a://{RAW_BUCKET}/iceberg_warehouse") \
    .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT) \
    .config("spark.hadoop.fs.s3a.access.key", S3_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", S3_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .getOrCreate()

def compile_training_batch():
    print("Starting ML Batch Pipeline...")

    # --- 2. Extract Candidate Pool from Postgres ---
    # We push down the basic filters directly to the database for efficiency
    query = """
    (SELECT 
        sg.generation_id, 
        sg.user_id, 
        sg.image_id, 
        sg.bbox::text as bbox, 
        sg.point_coords::text as point_coords, 
        sg.ml_suggested_mask, 
        sg.user_saved_mask, 
        sg.s3_sticker_key, 
        sg.processing_time_ms, 
        sg.num_tries, 
        sg.edited_pixels, 
        sg.generated_at
    FROM sticker_generations sg
    JOIN users u ON sg.user_id = u.user_id
    WHERE sg.saved = TRUE 
    AND sg.used_for_training = FALSE
    AND u.ml_training_opt_in = TRUE
    AND sg.edited_pixels < 2000) AS training_candidates
    """

    df_candidates = spark.read.format("jdbc") \
        .option("url", POSTGRES_URI) \
        .option("driver", "org.postgresql.Driver") \
        .option("dbtable", query) \
        .option("user", POSTGRES_USER) \
        .option("password", POSTGRES_PASSWORD) \
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
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lakehouse.ml_datasets")

    table_name = "lakehouse.ml_datasets.training_data"

    # 2. Smart Write: Create the table on the first run, append on future runs
    if spark.catalog.tableExists(table_name):
        print(f"Found existing Iceberg table. Appending {df_filtered.count()} records...")
        df_filtered.writeTo(table_name).append()
    else:
        print(f"First run detected! Creating Iceberg table and writing {df_filtered.count()} records...")
        df_filtered.writeTo(table_name).create()

    print("Batch successfully committed to Iceberg!")

    # --- 5. State Update: Mark as Used in Postgres ---
    # Extract the IDs that were just selected to update the source database
    processed_ids = [row.generation_id for row in df.collect()]

    if processed_ids:
        # 1. Open the connection
        conn = psycopg2.connect(
            host="postgres", database="sticker_gen", user=POSTGRES_USER, password=POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        
        # 2. Update the records
        # By passing tuple(processed_ids), psycopg2 safely formats it as: IN (10, 12, 13)
        cur.execute(
            "UPDATE sticker_generations SET used_for_training = TRUE WHERE generation_id IN %s",
            (tuple(processed_ids),) 
        )
        
        # 3. Commit and close
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"Successfully marked {len(processed_ids)} records as 'used_for_training' in Postgres.")

if __name__ == "__main__":
    compile_training_batch()
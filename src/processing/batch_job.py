from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, stddev, count, min, max, col, to_date
import os

# --- Konfiguration ---
DELTA_INPUT_PATH = os.getenv("DELTA_INPUT_PATH", "data/processed/stream_delta/")
AGG_FEATURE_PATH = os.getenv("AGG_FEATURE_PATH", "data/processed/batch_features/")

# --- Spark Session ---
spark = SparkSession.builder \
    .appName("PredictiveMaintenanceStream") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

# --- Lade Daten aus Delta Lake ---
stream_data = spark.read.format("delta").load(DELTA_INPUT_PATH)

# --- Tagesbasierte Feature-Aggregation ---
daily_features = stream_data \
    .withColumn("date", to_date("window_end")) \
    .groupBy("vehicle_id", "date") \
    .agg(
        avg("avg_rpm").alias("mean_rpm"),
        stddev("avg_rpm").alias("std_rpm"),
        avg("avg_temp").alias("mean_temp"),
        stddev("avg_temp").alias("std_temp"),
        count("error_count").alias("error_count_sum"),
        max("error_count").alias("error_count_max")
    )

# --- Speichere aggregierte Features als neue Delta-Tabelle ---
daily_features.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(AGG_FEATURE_PATH)

print("[INFO] Aggregated features written to:", AGG_FEATURE_PATH)

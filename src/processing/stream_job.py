from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window, avg, count
from pyspark.sql.types import StructType, IntegerType, DoubleType, StringType
import os

# --- Konfiguration über Umgebungsvariablen ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "vehicle_events")
DELTA_OUTPUT_PATH = os.getenv("DELTA_OUTPUT_PATH", "/app/data/processed/stream_delta")

# --- Spark Session mit Delta-Unterstützung ---
spark = SparkSession.builder \
    .appName("PredictiveMaintenanceStream") \
    .config("spark.sql.shuffle.partitions", 2) \
    .config("spark.sql.streaming.schemaInference", True) \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

# --- Schema der eingehenden JSON-Daten ---
event_schema = StructType() \
    .add("timestamp", StringType()) \
    .add("vehicle_id", IntegerType()) \
    .add("rpm", DoubleType()) \
    .add("engine_temp", DoubleType()) \
    .add("mileage", DoubleType()) \
    .add("error_code", IntegerType())

# --- Kafka-Daten lesen ---
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC_NAME) \
    .option("startingOffsets", "latest") \
    .load()

# --- JSON parsen ---
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), event_schema).alias("data")) \
    .select("data.*") \
    .withColumn("timestamp", to_timestamp("timestamp"))

# --- Rolling Window Feature Engineering ---
features = parsed_stream \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy(
        col("vehicle_id"),
        window(col("timestamp"), "1 minute", "30 seconds")
    ) \
    .agg(
        avg("rpm").alias("avg_rpm"),
        avg("engine_temp").alias("avg_temp"),
        count("error_code").alias("error_count")
    ) \
    .select(
        "vehicle_id",
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        "avg_rpm",
        "avg_temp",
        "error_count"
    )

# --- Daten als Delta Stream schreiben ---
query = features.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", f"{DELTA_OUTPUT_PATH}/_checkpoints") \
    .start(DELTA_OUTPUT_PATH)

query.awaitTermination()

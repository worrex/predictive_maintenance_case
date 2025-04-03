import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from pyspark.sql import SparkSession

# --- Pfade & MLflow Setup ---
AGG_FEATURE_PATH = os.getenv("AGG_FEATURE_PATH", "data/processed/batch_features/")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPERIMENT_NAME = "predictive_maintenance"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Spark: Lade aggregierte Features aus Delta Lake ---
spark = SparkSession.builder \
    .appName("TrainPredictiveModel") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .getOrCreate()

df_spark = spark.read.format("delta").load(AGG_FEATURE_PATH)

# --- Konvertiere zu Pandas ---
df = df_spark.toPandas()

# --- Zielvariable simulieren (Dummy für Demo-Zwecke) ---
df["label"] = (df["error_count_sum"] > 2).astype(int)

# --- Features & Ziel trennen ---
features = ["mean_rpm", "std_rpm", "mean_temp", "std_temp", "error_count_sum", "error_count_max"]
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# --- Training + Logging ---
with mlflow.start_run(run_name="rf_run"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Logging
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_params(model.get_params())
    mlflow.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    mlflow.sklearn.log_model(model, "rf_model")

    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"[INFO] Model logged to MLflow (experiment: {EXPERIMENT_NAME})")


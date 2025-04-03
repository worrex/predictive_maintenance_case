import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

# --- Konfiguration ---
AGG_FEATURE_PATH = os.getenv("AGG_FEATURE_PATH", "data/processed/batch_features/")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPERIMENT_NAME = "predictive_maintenance"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Spark Session starten ---
spark = SparkSession.builder \
    .appName("EvaluatePredictiveModel") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .getOrCreate()

# --- Daten laden ---
df_spark = spark.read.format("delta").load(AGG_FEATURE_PATH)
df = df_spark.toPandas()

# --- Zielvariable rekonstruieren ---
df["label"] = (df["error_count_sum"] > 2).astype(int)

features = ["mean_rpm", "std_rpm", "mean_temp", "std_temp", "error_count_sum", "error_count_max"]
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# --- Letztes Modell aus MLflow laden ---
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
latest_run = sorted(client.search_runs(experiment.experiment_id, order_by=["start_time desc"]), key=lambda x: x.start_time, reverse=True)[0]
model_uri = f"runs:/{latest_run.info.run_id}/rf_model"
model = mlflow.sklearn.load_model(model_uri)

# --- Vorhersagen & Evaluation ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_test, y_proba)

# --- Ergebnisse anzeigen ---
print("[INFO] Modellbewertung abgeschlossen:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc:.4f}")

# --- Optional: Ergebnisse in MLflow loggen ---
with mlflow.start_run(run_id=latest_run.info.run_id):
    mlflow.log_metric("roc_auc", roc_auc)

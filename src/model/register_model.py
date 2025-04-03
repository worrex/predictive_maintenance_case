import os
import mlflow
from mlflow.tracking import MlflowClient

# --- Konfiguration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "predictive_maintenance"
MODEL_NAME = "rf_predictive_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# --- Suche nach dem neuesten Modell-Run mit besten Metriken ---
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.f1_score DESC"],
    max_results=1
)

if not runs:
    raise ValueError("Kein gültiges Modellrun gefunden!")

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/rf_model"

print(f"[INFO] Bestes Modell gefunden: Run ID = {run_id}")

# --- Modell registrieren (wird Version 1, 2, ...) ---
result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"[INFO] Modell registriert unter dem Namen: {MODEL_NAME}")

# --- Automatisch auf 'Production' setzen (optional, Vorsicht im echten Prod!) ---
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"[INFO] Modell-Version {result.version} wurde auf 'Production' gesetzt.")

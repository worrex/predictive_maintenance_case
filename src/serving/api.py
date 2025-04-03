from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import os
import pandas as pd
import time
import requests

# --- Konfiguration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MODEL_NAME = os.getenv("MODEL_NAME", "rf_predictive_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")


# --- Warten, bis MLflow erreichbar ist ---
def wait_for_mlflow(host: str = "mlflow", port: int = 5050, timeout: int = 60):
    url = f"http://{host}:{port}/api/2.0/mlflow/experiments/list"
    print("[INFO] Waiting for MLflow to become available...")
    for i in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("[INFO] MLflow is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise Exception(f"[ERROR] MLflow did not respond after {timeout} seconds")


# --- Setup ---
wait_for_mlflow(
    host=MLFLOW_TRACKING_URI.replace("http://", "").split(":")[0],
    port=int(MLFLOW_TRACKING_URI.split(":")[-1])
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

app = FastAPI(
    title="Predictive Maintenance API",
    description="REST-API zur Vorhersage von Fahrzeugwartung basierend auf Sensordaten",
    version="1.0.0"
)


# --- Eingabe-Datenmodell ---
class InputData(BaseModel):
    vehicle_id: int
    mean_rpm: float
    std_rpm: float
    mean_temp: float
    std_temp: float
    error_count_sum: int
    error_count_max: int


# --- Root-Route ---
@app.get("/")
def read_root():
    return {"message": "Predictive Maintenance API is running."}



# --- Inferenz-Endpoint ---
@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {
        "vehicle_id": data.vehicle_id,
        "prediction": int(prediction[0]),  # 1 = Wartung empfohlen
    }

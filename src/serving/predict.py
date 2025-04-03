import requests
import json
import os

# --- API-Konfiguration ---
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")

# --- Beispielpayload (kann später auch aus CSV, Kafka etc. kommen) ---
sample_payload = {
    "vehicle_id": 101,
    "mean_rpm": 2450,
    "std_rpm": 180,
    "mean_temp": 87.2,
    "std_temp": 3.1,
    "error_count_sum": 4,
    "error_count_max": 3
}

def run_prediction(payload: dict):
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        print(f"[✅] Vorhersage für Fahrzeug {result['vehicle_id']}: {result['prediction']} (1 = Wartung empfohlen)")
    else:
        print(f"[❌] Fehler beim Abruf der Vorhersage: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print(f"[INFO] Sende Beispielvorhersage an {API_URL}")
    run_prediction(sample_payload)

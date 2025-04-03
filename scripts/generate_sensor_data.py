import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Konfiguration
n_samples = 1000
n_vehicles = 10
output_path = "../data/raw/raw_sensors.csv"

start_time = datetime.now()

data = []

for i in range(n_samples):
    timestamp = start_time + timedelta(seconds=i * 10)
    vehicle_id = np.random.randint(1, n_vehicles + 1)
    rpm = np.random.normal(2500, 300)
    temp = np.random.normal(85, 5)
    mileage = np.random.normal(150000, 20000)
    error_code = int(np.random.rand() < 0.02)  # 2% Fehlerrate

    data.append([timestamp.isoformat(), vehicle_id, rpm, temp, mileage, error_code])

df = pd.DataFrame(data, columns=[
    "timestamp", "vehicle_id", "rpm", "engine_temp", "mileage", "error_code"
])

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"[INFO] Sensor-Dataset mit {n_samples} Einträgen gespeichert unter: {output_path}")

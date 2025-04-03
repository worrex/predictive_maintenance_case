import json
import time
import pandas as pd
from kafka import KafkaProducer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "vehicle_events")
DATA_PATH = os.getenv("SIMULATION_DATA_PATH", "data/raw/raw_sensors.csv")
SLEEP_INTERVAL = float(os.getenv("SIMULATION_INTERVAL", 0.2))  # seconds between events

def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=str.encode
    )

def simulate_stream(producer, topic, data_path, interval):
    df = pd.read_csv(data_path)

    for idx, row in df.iterrows():
        event = row.to_dict()
        key = str(event.get("vehicle_id", "default"))
        producer.send(topic, key=key, value=event)
        print(f"[INFO] Sent event {idx + 1}/{len(df)} to topic '{topic}'")
        time.sleep(interval)

    producer.flush()
    print("[INFO] Stream simulation completed.")


if __name__ == "__main__":
    print(f"[INFO] Starting stream simulation to topic '{TOPIC_NAME}'...")
    producer = create_producer()
    simulate_stream(producer, TOPIC_NAME, DATA_PATH, SLEEP_INTERVAL)

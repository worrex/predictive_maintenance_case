#!/bin/bash

echo "🚀 Starte Predictive Maintenance MVP..."

echo "1️⃣ Starte Docker Compose..."
docker compose -f docker-compose.yml up --build -d

# Warte auf Kafka-Verfügbarkeit
echo "⏳ Warte auf Kafka..."
until docker compose exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list > /dev/null 2>&1; do
  sleep 2
done
echo "✅ Kafka ist verfügbar!"

echo "3️⃣ Starte Kafka-Datenstrom..."
python3 src/ingestion/stream_simulator.py &

sleep 5
# Warte auf Spark
echo "⏳ Warte auf Spark..."
until docker compose exec spark ls /app > /dev/null 2>&1; do
  sleep 2
done
echo "✅ Spark ist verfügbar!"

# Gib Spark noch einen Moment nach dem Start
sleep 3

echo "⚙️  Starte Spark Streaming Job..."
docker compose exec spark spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 \
  /app/src/processing/stream_job.py


sleep 5
echo "5️⃣ Starte Tagesaggregation..."
docker compose exec spark spark-submit /app/src/processing/batch_job.py

echo "6️⃣ Trainiere Modell..."
docker compose exec spark python3 /app/src/model/train_model.py > logs/train_model.log 2>&1

echo "7️⃣ Evaluiere Modell..."
docker compose exec spark python3 /app/src/model/evaluate_model.py

echo "8️⃣ Registriere bestes Modell..."
docker compose exec spark python3 /app/src/model/register_model.py

echo "9️⃣ Teste API-Vorhersage..."
python3 src/serving/predict.py

# Öffne UIs automatisch im Browser (macOS, Linux mit xdg-open oder Windows WSL-kompatibel)
MLFLOW_UI="http://localhost:5050"
FASTAPI_UI="http://localhost:8000/docs"

echo "🌐 Öffne MLflow UI: $MLFLOW_UI"
echo "🌐 Öffne FastAPI UI: $FASTAPI_UI"

if command -v open &> /dev/null; then
  open $MLFLOW_UI
  open $FASTAPI_UI
elif command -v xdg-open &> /dev/null; then
  xdg-open $MLFLOW_UI
  xdg-open $FASTAPI_UI
fi

echo "✅ MVP vollständig aktiv!"

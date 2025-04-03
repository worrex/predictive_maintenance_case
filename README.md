# Predictive Maintenance Case

Ein vollständiges, produktionsnahes End-to-End-Projekt zur Echtzeit-Wartungsvorhersage auf Basis von Fahrzeugdaten.  
Technologien: Spark, Kafka, Delta Lake, Scikit-learn, MLflow, FastAPI, Docker.

---

## Ziel

Vorhersage von Wartungsbedarf (Predictive Maintenance) aus Echtzeit-Sensordaten einer Fahrzeugflotte.

---

## Architektur

- **Kafka**: Sensordaten-Streaming
- **Spark Structured Streaming**: Feature-Engineering in Echtzeit
- **Delta Lake**: Speicherung aggregierter Zeitfensterdaten
- **Scikit-learn + MLflow**: Modelltraining & Tracking
- **FastAPI**: REST-API für Inferenz
- **Docker Compose**: Orchestrierung lokaler Dev-Umgebung

---

masterdaten waren auch dabei
ich war für die aufsetzung der pipeline zuständig 


## 🚀 Setup (lokal)

### 1. Klone das Repository & erstelle `.env`

```bash
git clone https://github.com/worrex/predictive_maintenance_case.git
cd predictive_maintenance_case
touch .env
# (Oder kopiere den Inhalt aus der Doku)

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner": "worrex",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="predictive_maintenance_training_pipeline",
    default_args=default_args,
    description="Tägliches Training für Predictive Maintenance Modell",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
) as dag:

    batch_features = BashOperator(
        task_id="run_batch_job",
        bash_command="spark-submit src/processing/batch_job.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python3 src/model/train_model.py",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="python3 src/model/evaluate_model.py",
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="python3 src/model/register_model.py",
    )

    # --- Task Dependencies ---
    batch_features >> train_model >> evaluate_model >> register_model

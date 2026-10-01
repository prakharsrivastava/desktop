# services/mlflow_service.py

import mlflow

def log_query_metrics(

    latency,
    precision,
    hallucination_rate

):

    with mlflow.start_run():

        mlflow.log_metric(

            "latency_ms",

            latency

        )

        mlflow.log_metric(

            "retrieval_precision",

            precision

        )

        mlflow.log_metric(

            "hallucination_rate",

            hallucination_rate

        )
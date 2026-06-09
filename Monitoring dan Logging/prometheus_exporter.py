from prometheus_client import start_http_server, Gauge
import psutil
import time
import json
import os

cpu_usage = Gauge(
    "cpu_usage",
    "CPU Usage Percentage"
)

memory_usage = Gauge(
    "memory_usage",
    "Memory Usage Percentage"
)

accuracy = Gauge(
    "model_accuracy",
    "Model Accuracy"
)

precision = Gauge(
    "model_precision",
    "Model Precision"
)

recall = Gauge(
    "model_recall",
    "Model Recall"
)

f1_score = Gauge(
    "model_f1_score",
    "Model F1 Score"
)

start_http_server(8000)

print("Prometheus Exporter running on port 8000...")

metrics_file = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Membangun_model",
    "model_metrics.json"
)

while True:

    cpu_usage.set(
        psutil.cpu_percent()
    )

    memory_usage.set(
        psutil.virtual_memory().percent
    )

    if os.path.exists(metrics_file):

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        accuracy.set(
            metrics["accuracy"]
        )

        precision.set(
            metrics["precision"]
        )

        recall.set(
            metrics["recall"]
        )

        f1_score.set(
            metrics["f1_score"]
        )

    time.sleep(5)
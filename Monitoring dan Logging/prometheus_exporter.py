from prometheus_client import start_http_server, Gauge, Counter
import psutil
import time

# =========================
# Metrics Model ML
# =========================

accuracy = Gauge(
    "model_accuracy",
    "Model Accuracy"
)

f1_score = Gauge(
    "model_f1_score",
    "Model F1 Score"
)

precision = Gauge(
    "model_precision",
    "Model Precision"
)

recall = Gauge(
    "model_recall",
    "Model Recall"
)

latency = Gauge(
    "prediction_latency",
    "Prediction Latency"
)

# =========================
# Metrics Monitoring Sistem
# =========================

cpu_usage = Gauge(
    "cpu_usage",
    "CPU Usage Percentage"
)

memory_usage = Gauge(
    "memory_usage",
    "Memory Usage Percentage"
)

total_predictions = Counter(
    "total_predictions",
    "Total Prediction Requests"
)

successful_predictions = Counter(
    "successful_predictions",
    "Total Successful Predictions"
)

failed_predictions = Counter(
    "failed_predictions",
    "Total Failed Predictions"
)

# =========================
# Start Prometheus Exporter
# =========================

start_http_server(8000)

print("Prometheus Exporter running on port 8000...")

while True:

    # =========================
    # Metrics hasil model
    # =========================

    accuracy.set(0.7843)
    f1_score.set(0.78)
    precision.set(0.79)
    recall.set(0.77)

    # =========================
    # Monitoring Sistem Nyata
    # =========================

    cpu_usage.set(psutil.cpu_percent())

    memory_usage.set(
        psutil.virtual_memory().percent
    )

    # =========================
    # Monitoring Inference
    # =========================

    total_predictions.inc()

    successful_predictions.inc()

    latency.set(0.15)

    time.sleep(5)
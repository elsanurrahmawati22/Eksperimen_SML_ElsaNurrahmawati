from prometheus_client import start_http_server, Gauge, Counter
import random
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

# Jalankan exporter
start_http_server(8000)

print("Prometheus Exporter running on port 8000...")

while True:

    # Metrics model
    accuracy.set(0.80)
    f1_score.set(0.56)
    precision.set(0.64)
    recall.set(0.51)
    latency.set(random.uniform(0.1, 1.0))

    # Metrics sistem
    cpu_usage.set(random.uniform(20, 80))
    memory_usage.set(random.uniform(30, 70))

    # Simulasi request prediksi
    total_predictions.inc()

    if random.random() > 0.1:
        successful_predictions.inc()
    else:
        failed_predictions.inc()

    time.sleep(5)
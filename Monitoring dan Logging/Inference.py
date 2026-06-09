import pandas as pd
import requests
import json
import os
import time

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge
)

# =========================
# Metrics Inference Nyata
# =========================

total_predictions = Counter(
    "total_predictions",
    "Total Prediction Requests"
)

successful_predictions = Counter(
    "successful_predictions",
    "Successful Predictions"
)

failed_predictions = Counter(
    "failed_predictions",
    "Failed Predictions"
)

prediction_latency = Gauge(
    "prediction_latency",
    "Prediction Latency"
)

# Export metrics ke port 8001
start_http_server(8001)

print("Inference Monitoring running on port 8001...")

csv_path = os.path.join(
    os.path.dirname(__file__),
    "telco_churn_preprocessed.csv"
)

df = pd.read_csv(csv_path)

sample = df.drop(
    "Churn",
    axis=1
).head(1)

while True:

    payload = {
        "inputs": sample.to_dict(
            orient="records"
        )
    }

    try:

        start_time = time.time()

        response = requests.post(
            "http://127.0.0.1:5001/invocations",
            headers={
                "Content-Type": "application/json"
            },
            data=json.dumps(payload)
        )

        latency = time.time() - start_time

        prediction_latency.set(
            latency
        )

        total_predictions.inc()

        if response.status_code == 200:

            successful_predictions.inc()

            print(
                "Prediksi berhasil:",
                response.json()
            )

        else:

            failed_predictions.inc()

            print(
                "Request gagal:",
                response.text
            )

    except Exception as e:

        failed_predictions.inc()

        print("Error:", e)

    time.sleep(5)
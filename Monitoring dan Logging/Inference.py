import pandas as pd
import requests
import json
import os

csv_path = os.path.join(
    os.path.dirname(__file__),
    "telco_churn_preprocessed.csv"
)

df = pd.read_csv(csv_path)

sample = df.drop("Churn", axis=1).head(1)

payload = {
    "inputs": sample.to_dict(orient="records")
}

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print("Data berhasil dimuat")
print(sample)

print("\nHasil Prediksi:")
print(response.json())
import pandas as pd
import os

csv_path = os.path.join(
    os.path.dirname(__file__),
    "telco_churn_preprocessed.csv"
)

df = pd.read_csv(csv_path)

sample = df.drop("Churn", axis=1).head(1)

print("Data berhasil dimuat")
print(sample)

print("\nInference berhasil dijalankan")
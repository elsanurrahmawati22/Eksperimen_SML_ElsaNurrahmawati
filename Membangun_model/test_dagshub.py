import dagshub
import mlflow

dagshub.init(
    repo_owner="elsanurrahmawati22",
    repo_name="telco-customer-churn-mlflow",
    mlflow=True
)

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.95)

print("Berhasil mengirim experiment ke DagsHub")
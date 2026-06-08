import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================
# DagsHub Configuration
# ==========================

dagshub.init(
    repo_owner="elsanurrahmawati22",
    repo_name="telco-customer-churn-mlflow",
    mlflow=True
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("telco_churn_preprocessed.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Hyperparameter Tuning
# ==========================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run():

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # ==========================
    # Metrics
    # ==========================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ==========================
    # Manual Parameter Logging
    # ==========================

    mlflow.log_param(
        "best_n_estimators",
        best_model.n_estimators
    )

    mlflow.log_param(
        "best_max_depth",
        best_model.max_depth
    )

    mlflow.log_param(
        "best_min_samples_split",
        best_model.min_samples_split
    )

    # ==========================
    # Manual Metric Logging
    # ==========================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # ==========================
    # Log Model
    # ==========================

    mlflow.sklearn.log_model(
        best_model,
        "random_forest_model"
    )

    # ==========================
    # Artifact 1
    # Confusion Matrix
    # ==========================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.savefig(
        "confusion_matrix.png"
    )

    plt.close()

    mlflow.log_artifact(
        "confusion_matrix.png"
    )

    # ==========================
    # Artifact 2
    # Classification Report
    # ==========================

    report = classification_report(
        y_test,
        y_pred
    )

    with open(
        "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    mlflow.log_artifact(
        "classification_report.txt"
    )

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nMetrics:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nBerhasil upload ke DagsHub!")
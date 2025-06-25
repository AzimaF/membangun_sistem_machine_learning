import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

# Prometheus monitoring
from prometheus_client import start_http_server, Counter

# 1. Konfigurasi MLflow ke DagsHub
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AzimaF/membangun_sistem_machine_learning.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzimaF"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a6fd2ef52a2527e3b4d345307141e1f2cf8e1182"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("RandomForest_Classifier")

# 2. Load dataset
df = pd.read_csv("Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 3. Preprocessing (standarisasi)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Inisialisasi model
model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)

# 5. Aktifkan MLflow autologging
mlflow.sklearn.autolog()

# 6. Training dan logging
with mlflow.start_run():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Manual log tambahan (opsional, melengkapi autolog)
    mlflow.log_param("manual_log", True)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_f1_score", train_f1)
    mlflow.log_metric("test_f1_score", test_f1)

    # Simpan model ke direktori
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    print(f"✅ Model trained and logged successfully.")
    print(f"🎯 Test Accuracy: {test_acc:.4f} | Test F1-Score: {test_f1:.4f}")

# 7. Monitoring dengan Prometheus
REQUEST_COUNT = Counter('model_training_requests_total', 'Total training requests')

if __name__ == "__main__":
    start_http_server(8000)
    REQUEST_COUNT.inc()
    print("📡 Prometheus metrics available at http://localhost:8000/metrics")

    # Jaga agar container tetap hidup
    while True:
        time.sleep(60)

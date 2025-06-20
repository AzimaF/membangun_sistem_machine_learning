# modelling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import os
import joblib

# 1. Set up MLflow tracking ke DagsHub
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AzimaF/membangun_sistem_machine_learning.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzimaF"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a6fd2ef52a2527e3b4d345307141e1f2cf8e1182"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("RandomForest_Classifier")

# 2. Load dan persiapan data
df = pd.read_csv("Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 3. Inisialisasi model
model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)

# 4. Logging ke MLflow
with mlflow.start_run():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Logging parameter
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("random_state", 42)

    # Logging metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_f1_score", train_f1)
    mlflow.log_metric("test_f1_score", test_f1)

    # Simpan model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.pkl")
    mlflow.log_artifact("models/random_forest_model.pkl")

    print(f"✅ Model trained successfully.")
    print(f"🎯 Test Accuracy: {test_acc:.4f} | Test F1-Score: {test_f1:.4f}")

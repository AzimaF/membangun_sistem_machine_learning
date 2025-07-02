import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# 1. Konfigurasi MLflow lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # default local URI
mlflow.set_experiment("RandomForest_Classifier_Local")

# 2. Load dataset
df = pd.read_csv("membangun_sistem_machine_learning-main/membangun_model/Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 3. Preprocessing (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Aktifkan autolog
mlflow.sklearn.autolog()

# 6. Training dan logging otomatis
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    print("✅ Model trained and autologged successfully.")

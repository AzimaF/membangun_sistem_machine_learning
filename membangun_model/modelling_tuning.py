# modelling_tuning.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib

# 🧪 Setup koneksi ke DagsHub MLflow
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AzimaF/membangun_sistem_machine_learning.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzimaF"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a6fd2ef52a2527e3b4d345307141e1f2cf8e1182"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("RandomForest_Tuning")

# 📂 Load dataset
df = pd.read_csv("Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 🔁 Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🎯 Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 🔍 Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15, 20],
    'random_state': [42]
}

# 🚀 Run eksperimen
with mlflow.start_run():
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # 📈 Evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 📝 Logging parameter dan metrik
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # 💾 Simpan model ke file dan log manual (hindari error DagsHub)
    joblib.dump(best_model, "best_random_forest_model.pkl")
    mlflow.log_artifact("best_random_forest_model.pkl")

    print("Best Params:", grid.best_params_)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

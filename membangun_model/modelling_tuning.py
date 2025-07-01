import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ============================
# 🔐 MLflow DagsHub Tracking
# ============================
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AzimaF/membangun_sistem_machine_learning.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzimaF"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a6fd2ef52a2527e3b4d345307141e1f2cf8e1182"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("RandomForest_Tuning")

# ============================
# 📂 Load Dataset
# ============================
df = pd.read_csv("membangun_sistem_machine_learning-main/membangun_model/Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

# ============================
# 🔁 Feature Scaling
# ============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# 🎯 Split Data
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# 🔍 Hyperparameter Grid
# ============================
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15, 20],
    'random_state': [42]
}

# ============================
# 🚀 Training + Logging
# ============================
with mlflow.start_run():

    grid = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # 📊 Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion Matrix (gunakan argmax jika multiclass tidak bisa di-ravel)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        fpr = fp / (fp + tn)
    except:
        specificity = None
        fpr = None

    # 📝 Log ke MLflow
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    if specificity is not None:
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("false_positive_rate", fpr)

    # 💾 Log model ke folder 'random_forest_model'
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # ✅ Output Info
    print("✅ Model dilatih dan dicatat di MLflow (DagsHub).")
    print(f"🎯 Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    if specificity is not None:
        print(f"📈 Specificity: {specificity:.4f} | FPR: {fpr:.4f}")

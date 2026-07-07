import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from mlflow.models.signature import infer_signature

# ============================
# 📂 Load Dataset
# ============================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

X = df.drop("label", axis=1)
y = df["label"]

# ============================
# 🔁 Preprocessing
# ============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# 🚀 MLflow Run
# ============================
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')

    # ========================
    # 📝 Log ke MLflow
    # ========================
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_f1_score", train_f1)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # ========================
    # 🔍 Signature & Input Example
    # ========================
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = pd.DataFrame(X_train[:5], columns=X.columns if isinstance(X, pd.DataFrame) else df.columns.drop("label"))

    # ========================
    # 💾 Log Model + File
    # ========================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.pkl")
    mlflow.log_artifact("models/random_forest_model.pkl")

    # ========================
    # ✅ Output Info
    # ========================
    print("✅ Model trained and logged to MLflow.")
    print(f"🎯 Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print("📦 Model saved to: models/random_forest_model.pkl")
    print("🔁 Run ID:", run.info.run_id)

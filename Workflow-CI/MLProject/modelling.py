import os
import time
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Crop_recommendation.csv")
    df = pd.read_csv(csv_path)
    return df

def train_and_log():
    # === 1. Load data
    df = load_data()
    X = df.drop("label", axis=1)
    y = df["label"]

    # === 2. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # === 3. Init model
    model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42)

    # === 4. MLflow run
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        # === Log param & metrics
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("test_f1_score", test_f1)

        # === Log model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = pd.DataFrame(X_train[:5], columns=df.columns.drop("label"))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

        # === Save model file
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/random_forest_model.pkl")
        mlflow.log_artifact("models/random_forest_model.pkl")

        print("✅ Model trained & logged to MLflow.")
        print(f"🎯 Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
        print("🔁 Run ID:", run.info.run_id)

if __name__ == "__main__":
    # Setup MLflow to DagsHub
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AzimaF/membangun_sistem_machine_learning.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "AzimaF"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "a6fd2ef52a2527e3b4d345307141e1f2cf8e1182"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("RandomForest_Classifier")

    train_and_log()

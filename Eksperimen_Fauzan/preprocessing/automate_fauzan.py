import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Konfigurasi manual scaler 
scaler = StandardScaler()
scaler.mean_ = np.array([50.5, 53.4, 48.1, 25.6, 71.5, 6.5, 103.5])
scaler.scale_ = np.array([37.0, 33.0, 50.6, 5.0, 22.3, 0.8, 55.0])
scaler.var_ = scaler.scale_ ** 2

# Load model terlatih
model_path = "random_forest_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di {model_path}. Pastikan file tersedia.")

model = joblib.load(model_path)

# Input dari pengguna
def input_user():
    print("\nSilakan masukkan nilai untuk masing-masing fitur:")
    N = float(input("N (Nitrogen): "))
    P = float(input("P (Phosphorus): "))
    K = float(input("K (Potassium): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH tanah: "))
    rainfall = float(input("Rainfall (mm): "))
    return [[N, P, K, temperature, humidity, ph, rainfall]]

# Proses prediksi
def predict_crop():
    user_input = input_user()
    scaled_input = scaler.transform(user_input)
    predicted_crop = model.predict(scaled_input)
    print("\n🌱 Jenis tanaman yang direkomendasikan:", predicted_crop[0])

    # Ekspor hasil ke CSV
    export_predictions_to_csv(user_input, predicted_crop)

# Ekspor hasil prediksi ke file CSV
def export_predictions_to_csv(data, predictions, filename="crop_prediction_result.csv"):
    columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df_predictions = pd.DataFrame(data, columns=columns)
    df_predictions['predicted_crop'] = predictions
    df_predictions.to_csv(filename, index=False)
    print(f"\n📁 Prediksi telah diekspor ke {filename}")

if __name__ == "__main__":
    predict_crop()
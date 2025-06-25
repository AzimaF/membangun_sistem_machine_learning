from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import time
import joblib

# Inisialisasi Flask app
app = Flask(__name__)

# simpan model di MLflow
model = joblib.load("models/random_forest_model.pkl")

# Hitung dan cetak latency tiap request
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    latency = (time.time() - request.start_time) * 1000
    print(f"Latency: {latency:.2f} ms")
    return response

# Endpoint prediksi
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        prediction = model.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

# Jalankan server Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from prometheus_client import start_http_server, Histogram, Counter
import time
from flask import Flask

# Definisikan metrik
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of incoming requests in seconds')
REQUEST_COUNT = Counter('request_count_total', 'Total number of HTTP requests')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

@app.route("/trigger")
def trigger():
    REQUEST_COUNT.inc()  # Tambah 1 untuk setiap request
    with REQUEST_LATENCY.time():  # Hitung waktu eksekusi
        time.sleep(0.1)  # Simulasi proses server
    return "Request recorded successfully!"

if __name__ == "__main__":
    # Jalankan Prometheus metrics server di port 8000
    start_http_server(8000)
    
    # Jalankan server Flask di port 5001
    app.run(host="0.0.0.0", port=5001)

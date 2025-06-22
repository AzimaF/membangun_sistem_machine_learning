# Menggunakan base image Python
FROM python:3.11-slim

# Direktori kerja di dalam container
WORKDIR /app

# Salin semua isi folder lokal ke container
COPY . .

# Install dependensi
RUN pip install --upgrade pip && \
    pip install mlflow scikit-learn pandas numpy dagshub
    pip install --no-cache-dir -r requirements.txt

# Jalankan script MLflow
CMD ["python", "modelling.py"]

# Menggunakan base image Python
FROM python:3.11

# Direktori kerja di dalam container
WORKDIR /app

# Salin semua isi folder lokal ke container
COPY . /app

# Install dependensi
RUN pip install --upgrade pip && \
    pip install mlflow scikit-learn pandas numpy dagshub
    pip install flask prometheus_client

# Expose port
EXPOSE 8000

# Jalankan script MLflow
CMD ["python", "prometheus_exporter.py"]

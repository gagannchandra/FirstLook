# FirstLook offline runtime (L2). Build: docker build -t firstlook -f docker/Dockerfile .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY src/__init__.py src/runtime.py src/features.py src/calibrate.py src/conformal.py src/benchmark.py src/
COPY app/ app/
COPY .streamlit/ .streamlit/
COPY models/bundles/ models/bundles/
COPY results/dashboard/ results/dashboard/
EXPOSE 8501
# Benchmark (offline proof):  docker run --rm --network none --cpus 2 --memory 2g firstlook python -m src.benchmark
# UI (port mapping needs a network; for the offline UI demo switch Wi-Fi off instead):
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--browser.serverAddress", "localhost"]

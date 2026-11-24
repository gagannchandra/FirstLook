# FirstLook offline runtime (L2). Build: docker build -t firstlook -f docker/Dockerfile .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 ORT_DISABLE_TELEMETRY=1
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# Only the runtime slice of the pipeline: features -> calibrate/conformal -> ONNX runtime + benchmark.
COPY src/__init__.py src/
COPY src/data/__init__.py src/data/features.py src/data/
COPY src/modelling/__init__.py src/modelling/calibrate.py src/modelling/conformal.py src/modelling/
COPY src/edge/__init__.py src/edge/runtime.py src/edge/benchmark.py src/edge/
COPY app/ app/
COPY .streamlit/ .streamlit/
COPY models/bundles/ models/bundles/
COPY results/dashboard/ results/dashboard/
EXPOSE 8501
# Benchmark (offline proof):  docker run --rm --network none --cpus 2 --memory 2g firstlook python -m src.edge.benchmark
# UI (port mapping needs a network; for the offline UI demo switch Wi-Fi off instead):
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "::", "--browser.serverAddress", "localhost"]

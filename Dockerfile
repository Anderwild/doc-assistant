from python:3.11-slim

run apt-get update && apt-get install -y --no-install-recommend \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
workdir /app

env HF_HOME=/app/cache/huggingface \
    PYTHONPATH=/app \
    PYTHONBUFFERED=1

copy rq.in
run pip install --no-cache-dir -r rq.in
copy src/ ./src/
copy .env .
expose 8000

cmd ["uvicorn","src.app:app","--host","0.0.0.0","--port","8000"]

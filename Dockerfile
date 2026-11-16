FROM python:3.11-slim
     2	
     3	RUN apt-get update && apt-get install -y --no-install-recommends \
     4	    build-essential \
     5	    && rm -rf /var/lib/apt/lists/*
     6	WORKDIR /app
     7	
     8	ENV HF_HOME=/app/cache/huggingface \
     9	    PYTHONPATH=/app \
    10	    PYTHONUNBUFFERED=1
    11	
    12	COPY rq.in .
    13	RUN pip install --no-cache-dir -r rq.in
    14	COPY src/ ./src/
    15	COPY .env .
    16	
    17	EXPOSE 8000
    18	
    19	CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
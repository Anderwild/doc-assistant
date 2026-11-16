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
    13	# faiss-cpu не имеет готовых wheel под Python 3.11 Ч ставим подсовместимую версию (дл€ 3.10)
    14	# torch фиксируем на CPU-build: иначе в slim-контейнер т€нетс€ CUDA-верси€ (~3 √Ѕ)
    15	RUN sed -i 's/^faiss-cpu$/faiss-cpu==1.8.0/; s/^torch$/torch==2.5.1+cpu/' rq.in && \
    16	    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r rq.in
    17	COPY src/ ./src/
    18	COPY .env .
    19	
    20	EXPOSE 8000
    21	
    22	CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
    23	
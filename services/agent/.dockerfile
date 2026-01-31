FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGENT_STATUS_TIME=0.2

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/agent/requirements.txt

COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    echo "Embedding model all-MiniLM-L6-v2 pre-downloaded to /root/.cache/chroma/onnx_models/"

COPY agent /app/agent

WORKDIR /app/agent

CMD ["python", "-m", "app.main"]
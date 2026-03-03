FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGENT_STATUS_TIME=0.2 \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        libnss3 \
        libgbm1 \
        libasound2 \
        libatk-bridge2.0-0 \
        libdrm2 \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && apt-get purge -y gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/agent/requirements.txt

COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    echo "SentenceTransformers model pre-downloaded"

RUN python -c "\
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; \
ef = ONNXMiniLM_L6_V2(); \
ef(['warmup']); \
print('ChromaDB ONNX embedding model pre-downloaded')" && \
    find /root/.cache -name '*MiniLM*' -type d | head -5

COPY agent /app/agent

WORKDIR /app/agent

CMD ["python", "-m", "app.main"]
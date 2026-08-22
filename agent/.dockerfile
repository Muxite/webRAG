FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGENT_STATUS_TIME=0.2 \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/agent/requirements.txt

# Install the Chromium browser used by the headless fallback (ConnectorBrowser).
# --with-deps pulls the required system libraries (libnss3, libgbm1, etc.).
RUN playwright install --with-deps chromium

COPY services/shared /app/shared
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

# The benchmark compose services (promptbench, ladder-benchmark) run as the invoking
# HOST uid so their result files are not root-owned. /root's default 0700 made the
# pre-downloaded embedding caches above unreachable for that uid, and every cell
# re-downloaded the 79 MB ONNX model instead. Only the directory bit is relaxed (the
# caches themselves are already world-readable) and nothing secret lives in /root.
RUN chmod 755 /root

WORKDIR /app/agent

CMD ["python", "-m", "app.main"]
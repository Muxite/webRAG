FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/agent/requirements.txt
RUN mkdir -p /app/model && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/app/model')"
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared
COPY agent /app/agent
WORKDIR /app/agent
ENV PYTHONPATH=/app
ENV AGENT_STATUS_TIME=0.2
CMD ["python", "-m", "app.main"]
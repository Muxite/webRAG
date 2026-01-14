FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
COPY metrics/requirements.txt /app/metrics/requirements.txt
RUN pip install --no-cache-dir -r /app/metrics/requirements.txt
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared
COPY metrics /app/metrics
WORKDIR /app/metrics
ENV PYTHONPATH=/app
CMD ["python", "-m", "app.main"]


FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared
WORKDIR /app
ENV PYTHONPATH=/app


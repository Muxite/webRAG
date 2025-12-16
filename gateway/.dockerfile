FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
RUN mkdir -p /app/gateway
COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements.txt
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared
COPY gateway /app/gateway
COPY special_api_keys.txt /app/gateway/special_api_keys.txt
WORKDIR /app/gateway
ENV PYTHONPATH=/app/gateway:/app
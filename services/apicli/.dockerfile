FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY apicli/requirements.txt /app/apicli/requirements.txt
RUN pip install --no-cache-dir -r /app/apicli/requirements.txt
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared
COPY apicli /app/apicli
WORKDIR /app/apicli
ENV PYTHONPATH=/app

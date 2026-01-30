FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements.txt
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared
COPY gateway /app/gateway
WORKDIR /app/gateway
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Set default command
CMD ["python", "-m", "app.main"]
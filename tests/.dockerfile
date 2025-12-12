FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Copy and install shared first (as a local package)
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared

# Install third-party deps for agent and gateway (include pytest)
COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/agent/requirements.txt

COPY gateway/requirements.txt /app/gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/gateway/requirements.txt

# Now copy the entire repo for tests
COPY . /app

# Ensure local packages/modules are importable
ENV PYTHONPATH=/app

# Default command (overridable by compose)
CMD ["pytest", "-vv"]

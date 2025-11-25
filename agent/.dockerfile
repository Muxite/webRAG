FROM python:3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \

WORKDIR /agent
COPY . .
RUN pip install --no-cache-dir .
ENV PYTHONPATH=/agent:/agent/shared
RUN find . -type d -name "__pycache__" -exec rm -r {} + || true
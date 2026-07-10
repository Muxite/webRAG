FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY connector_api/requirements.txt /app/connector_api/requirements.txt
RUN pip install --no-cache-dir -r /app/connector_api/requirements.txt

# Chromium powers the headless fallback (ConnectorBrowser) used by /visit on
# 401/403 bot-blocks. --with-deps pulls the required system libraries.
RUN playwright install --with-deps chromium

COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared

# The connector code lives in the agent service; import it directly rather than
# duplicating it. Only the connector_* modules are exercised here.
COPY agent /app/agent
COPY connector_api /app/connector_api

WORKDIR /app/connector_api

EXPOSE 13375

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "13375"]

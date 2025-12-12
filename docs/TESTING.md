### Testing Overview

`docker compose up agent-test --build`
#### What Agent Tests Cover
- Worker consumes a task and emits `accepted`, `started`, terminal status.
- Agent connectors (LLM/search/http/chroma) are unit-tested with mocks.
- Prompt building, tick output parsing, agent loop basics.

`docker compose up gateway-test --build`
#### What Gateway Tests Cover
- API key enforcement on `/tasks`.
- Task submission publishes a message to the configured input queue.
- Status flow reads RabbitMQ status messages and updates storage.
- E2E test drives the FastAPI app in-process and publishes `accepted`/`completed` statuses via RabbitMQ to observe gateway state changes.

#### What Shared Tests Cover
- RabbitMQ connector queue declaration and publish/consume basics.
- Retry and utility helpers as used by connectors.

#### Blind Spots
- No full end-to-end with a running agent container and gateway together.
- No load, soak, or chaos tests.
- No performance profiling or latency budgeting.
- No coverage of FastAPI lifespan deprecations; startup/shutdown events still used.

### Testing Guide
This project provides layered tests to validate the RabbitMQ-based workflow from small integration checks to
end-to-end scenarios with multiple containers.

Key points:
- No API mocks for RabbitMQ: tests use a real RabbitMQ broker.

#### Prerequisites
- Python 3.10+
- Docker (for running RabbitMQ and optional multi-container e2e)
- `pytest`

#### Run tests

- Inside containers via compose (preferred)..
  - `docker compose up --build gateway-test`
  - `docker compose up --build agent-test`

#### Multi-container E2E

- The compose services start a real RabbitMQ and Redis. Tests exercise them across containers.

#### Logging

Pytest is configured to stream logs at INFO level. On failures, inspect container logs as needed:
```
docker logs euglena_rabbitmq --tail=200
```

Logs:
```
docker logs euglena-rabbitmq-1 --tail=200 -f
```

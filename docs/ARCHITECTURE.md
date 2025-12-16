# Architecture

## Components

- **`agent/`**: Worker that consumes tasks from RabbitMQ, executes agent logic, and writes status to Redis
- **`gateway/`**: FastAPI service that accepts tasks, publishes to RabbitMQ, and serves status from Redis
- **`shared/`**: Common utilities (config, connectors, models, retry helpers)

## Runtime Flow

1. Client submits task via gateway `/tasks` endpoint
2. Gateway stores `pending` state in Redis and publishes `TaskEnvelope` to RabbitMQ
3. Agent worker consumes task and runs ticked loop:
   - Emits `accepted` -> `started` -> periodic `in_progress` -> `completed`/`error`
   - Writes latest state to Redis on each transition
4. Gateway serves `/tasks/{id}` by reading from Redis (no RabbitMQ consumption needed)

## Key Modules

**Agent Worker** (`interface_agent.py`): Consumes RabbitMQ, runs `Agent`, publishes status, maintains presence

**Agent Core** (`agent.py`): Ticked loop with LLM calls, action execution (search/visit/think/exit), Chroma storage

**Gateway Service** (`gateway_service.py`, `api.py`): FastAPI with API key auth, Redis-backed task storage


## CLI Modules
**Agent CLI** (`basic_cli.py`): Basic CLI for testing agent locally.

**API CLI** (`apicli.py`): API caller container that allows users to submit tasks via API.
It can add tasks using an API key, and allow manual status checks using the correllation id.


## Status States

- `pending` → initial state
- `accepted` → task acknowledged
- `started` → agent constructed
- `in_progress` → periodic heartbeat with tick info
- `completed` → success with result
- `error` → failure with error details

## Configuration

Environment variables (see `.env`):
- `RABBITMQ_URL`, `REDIS_URL`, `CHROMA_URL`, `GATEWAY_URL`
- `MODEL_API_URL`, `OPENAI_API_KEY`, `SEARCH_API_KEY`
- `AGENT_STATUS_TIME`, `AGENT_INPUT_QUEUE`, `AGENT_STATUS_QUEUE`

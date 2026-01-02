# Architecture

## Components

- **`frontend/`**: React web interface with Supabase authentication, task submission UI, and real-time status polling
- **`agent/`**: Worker that consumes tasks from RabbitMQ, executes agent logic, and writes status to Redis
- **`gateway/`**: FastAPI service that accepts tasks, publishes to RabbitMQ, and serves status from Redis
- **`shared/`**: Common utilities (config, connectors, models, retry helpers)

## Runtime Flow

1. Client submits task via gateway `/tasks` endpoint with Supabase JWT token
2. Gateway validates authentication and quota, stores `pending` state in Redis and publishes `TaskEnvelope` to RabbitMQ
3. Agent worker consumes task and runs ticked loop:
   - Emits `accepted` → `started` → periodic `in_progress` → `completed`/`error`
   - Writes latest state to Redis on each transition
   - Publishes status updates to RabbitMQ
4. Gateway serves `/tasks/{id}` by reading from Redis

## Key Modules

**Agent Worker** (`interface_agent.py`): Consumes RabbitMQ tasks, initializes and reuses connectors via dependency injection, verifies dependencies ready before consuming, runs Agent per task, publishes status, maintains presence, handles graceful shutdown

**Agent Core** (`agent.py`): Ticked loop with LLM calls, action execution (search/visit/think/exit), ChromaDB storage/retrieval, connectors injected for reuse

**Gateway Service** (`gateway_service.py`, `api.py`): FastAPI with Supabase auth, Redis task storage, RabbitMQ publishing, per-user quota enforcement

## Dependency Injection Pattern

Connectors (LLM, Search, HTTP, Chroma) created once in InterfaceAgent, reused across mandates. Reduces overhead, improves utilization, maintains persistent connections, ensures readiness before processing.

## Status States

- `pending` → initial state
- `accepted` → task acknowledged
- `started` → agent constructed
- `in_progress` → periodic heartbeat with tick info
- `completed` → success with result
- `error` → failure with error details

## Configuration

Environment variables (see `services/.env` and `services/keys.env`):
- `RABBITMQ_URL`, `REDIS_URL`, `CHROMA_URL`, `GATEWAY_URL`
- `MODEL_API_URL`, `OPENAI_API_KEY`, `SEARCH_API_KEY`
- `AGENT_STATUS_TIME`, `AGENT_INPUT_QUEUE`, `AGENT_STATUS_QUEUE`
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`

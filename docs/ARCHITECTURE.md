### Euglena Architecture
Three parts:
- `agent/`: worker that pulls tasks from RabbitMQ, runs the agent, pushes status.
- `gateway/`: FastAPI that accepts tasks, stores state, publishes to RabbitMQ, consumes status.
- `shared/`: common config, models, connectors for RabbitMQ/Redis, retry helpers.

---

### Runtime Flow
1. Client calls gateway `/tasks` with `mandate` (and optional `max_ticks`).
2. Gateway stores a `pending` record and publishes `TaskEnvelope` to `AGENT_INPUT_QUEUE` (`agent.mandates` default) with `correlation_id=task_id`.
3. Agent worker consumes the queue, replies with status envelopes to `AGENT_STATUS_QUEUE` (`agent.status` default).
4. Worker runs the ticked `Agent` loop, emits `accepted`, `started`, periodic `in_progress`, then `completed` or `error`.
5. Gateway consumes status updates and folds them into storage so `/tasks/{id}` reflects latest state.
6. Workers also write presence heartbeats into Redis for discoverability.

---

### Shared Pieces
- `shared/connector_config.py`: env loader for Redis, RabbitMQ, Chroma, LLM/search keys, retry timing, queues, tick limits.
- `shared/message_contract.py`: keys/enums/models for queue payloads (`TaskEnvelope`, `StatusEnvelope`, `StatusType`, `TaskState`, `KeyNames`).
- `shared/connector_rabbitmq.py`: aio-pika wrapper; declares queues, `publish_task`, `publish_status`, `consume_status_updates`.
- `shared/connector_redis.py`: async Redis client with retries and JSON helpers; tolerates missing Redis.
- `shared/worker_presence.py`: maintains Redis sets/keys for worker liveness using `status_time`.
- `shared/models.py`: FastAPI request/response models for gateway storage.

---

### Agent Worker (`agent/app/agent_worker.py`)
- Connects to RabbitMQ, consumes tasks, validates payloads.
- Publishes `accepted`/`started`, starts heartbeat loop sending `in_progress` with tick and counters.
- Runs `Agent` and publishes `completed` with result or `error` on failure.
- Uses `WorkerPresence` to emit Redis heartbeats; shuts down cleanly on stop.

### Agent Core (`agent/app/agent.py`)
- Ticked loop that builds prompts, calls LLM, parses `TickOutput`, performs actions (`search`, `visit`, `think`, `exit`), stores/retrieves context from Chroma, and returns a final result via `FinalOutputBuilder`.
- Tracks `current_tick`, `max_ticks`, and memory lists that heartbeats read.

### Agent Connectors
- `connector_llm.py`, `connector_search.py`, `connector_http.py`, `connector_chroma.py` handle external calls using `ConnectorConfig`.
- Helpers: `prompt_builder.py`, `tick_output.py`, `observation.py`, `final_output_builder.py`.
- Entrypoints: `basic_cli.py`, `main.py` for manual or service runs.

---

### Gateway Service (`gateway/app/gateway_service.py`, `gateway/app/api.py`)
- FastAPI app with API key check. Startup wires `GatewayService.start`, shutdown stops it.
- `create_task` persists `TaskRecord` to `RedisTaskStorage`, publishes task envelope to RabbitMQ.
- Background consumer maps `StatusEnvelope` → `TaskState` and updates storage.
- `/tasks/{id}` returns latest stored state; unknown ids return a well-formed placeholder (`status="unknown"`, `error="not found"`).
- Depends on `ConnectorConfig` and `ConnectorRabbitMQ`; quota hooks are optional and currently inactive.

---

### Status Semantics
- `accepted` → task acknowledged.
- `started` → agent constructed.
- `in_progress` → periodic heartbeat, includes `tick`, `max_ticks`, counters.
- `completed` → success result payload.
- `error` → failure payload.
Timing uses `AGENT_STATUS_TIME` (`status_time` in config).

### Presence
- Redis set `workers:agent` and expiring keys `worker:agent:{id}` (~3 × `status_time` TTL).
- Presence is best-effort and does not block workers when Redis is missing.

---

### Configuration Quick List
- `RABBITMQ_URL`, `AGENT_INPUT_QUEUE`, `AGENT_STATUS_QUEUE`, `AGENT_STATUS_TIME`
- `REDIS_URL`, `CHROMA_URL`
- `MODEL_API_URL`, `MODEL_NAME`, `OPENAI_API_KEY`, `SEARCH_API_KEY`
- `DEFAULT_DELAY`, `DEFAULT_TIMEOUT`, `JITTER_SECONDS`, `DAILY_TICK_LIMIT`

---

### Testing Notes
- `gateway/tests` use httpx `ASGITransport` with manual lifespan; status flow tests publish status messages directly to RabbitMQ to drive gateway state. Queues are declared before publish/consume and the service waits for connector readiness during startup.
- `agent/tests` cover worker orchestration and connector behavior with mocked externals.
- `shared/tests` cover RabbitMQ/Redis helpers.

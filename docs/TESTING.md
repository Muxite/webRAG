# Testing

## Running Tests

```bash
# Agent tests
docker compose run agent-test

# Gateway tests  
docker compose run gateway-test

# Shared tests
docker compose run shared-test
```

## Test Coverage

**Agent Tests** (`agent/tests/`):
- Worker orchestration (task consumption, status emission)
- Connector unit tests (LLM, search, HTTP, Chroma) with mocks
- Agent loop logic, prompt building, tick parsing

**Gateway Tests** (`gateway/tests/`):
- Supabase authentication enforcement
- Task submission and RabbitMQ publishing
- Status retrieval from Redis
- End-to-end with live agent container

**Shared Tests** (`shared/tests/`):
- RabbitMQ connector (queue declaration, publish/consume)
- Redis connector and retry helpers

## Test Architecture

- Tests use real RabbitMQ/Redis containers (no mocks)
- Gateway E2E runs FastAPI in-process with ASGITransport
- Agent E2E tests against live agent container
- Status flow validated via Redis (no RabbitMQ status consumption)

## Limitations

- No full multi-container E2E with gateway + agent
- No load/chaos testing
- No performance profiling

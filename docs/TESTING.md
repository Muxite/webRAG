# Testing

## Running Tests

```bash
cd services
docker compose --profile test up agent-test
docker compose --profile test up gateway-test
docker compose --profile test up shared-test
```

## Test Coverage

**Agent Tests** (`agent/tests/`): Worker orchestration, connector unit tests with mocks, agent loop logic, dependency injection patterns, lifecycle handling, readiness checks

**Gateway Tests** (`gateway/tests/`): Supabase auth enforcement, task submission, status retrieval, end-to-end with live agent

**Shared Tests** (`shared/tests/`): RabbitMQ connector, Redis connector, retry helpers

## Test Architecture

Uses real RabbitMQ/Redis containers. Gateway E2E runs FastAPI in-process. Agent E2E tests against live container. Pytest fixtures reduce complexity. Tests focus on single behaviors.

## Test Patterns

**Fixtures**: Common setup moved to reusable pytest fixtures
**Dependency Injection**: Tests create connectors and inject into Agent
**Mocking**: External APIs mocked, infrastructure uses real services
**Cleanup**: Tests properly clean up connections and tasks

## Limitations

- No full multi-container E2E with gateway + agent
- No load/chaos testing
- No performance profiling

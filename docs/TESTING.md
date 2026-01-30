# Testing

## Architecture Compliance

All tests are designed to verify compliance with the system architecture principles (see [ARCHITECTURE.md](ARCHITECTURE.md)):

### Core Architecture Tests

1. **Worker Isolation Tests**
   - `test_agent_only_writes_to_redis`: Verifies agents only write to Redis, never Supabase
   - `test_agent_redis_registration_and_task_completion`: Verifies workers register in Redis and update status in Redis only

2. **Gateway as Mediator Tests**
   - `test_gateway_writes_to_redis_and_supabase_on_creation`: Verifies gateway writes to both Redis and Supabase on task creation
   - `test_gateway_reads_from_supabase_as_source_of_truth`: Verifies gateway prioritizes Supabase as source of truth

3. **Redis Sync Tests**
   - `test_gateway_syncs_redis_to_supabase`: Verifies gateway syncs Redis updates to Supabase
   - `test_redis_sync_after_worker_update`: Integration test verifying full sync flow from worker → Redis → Gateway → Supabase

4. **Worker Registration Tests**
   - `test_workers_register_in_redis`: Verifies workers register in Redis and gateway reads from Redis
   - `test_agent_count`: Verifies gateway reads worker count from Redis

### Test Organization

Tests are organized to reflect architectural layers:
- **Agent Tests**: Verify worker isolation and Redis-only writes
- **Gateway Tests**: Verify dual writes, sync logic, and Supabase as source of truth
- **Integration Tests**: Verify end-to-end flow respecting all architecture principles

# Testing

## Running Tests

### Automated Test Runner (Recommended)

```bash
cd services
python ../scripts/run_tests.py
```

This script will:
1. Build all containers
2. Start infrastructure services (rabbitmq, redis, chroma) and wait for RabbitMQ to be ready (~2 minutes on first run)
3. Run tests in order: agent-test → gateway-test → integration-test
4. Stop on first failure
5. Keep infrastructure services running between test runs (saves ~2 minutes per test run)
6. Clean up only test and application containers between runs

#### Skipping Test Suites

You can skip specific test suites using command-line arguments:

```bash
# Skip agent tests
python ../scripts/run_tests.py --skip-agent

# Skip gateway tests
python ../scripts/run_tests.py --skip-gateway

# Skip integration tests
python ../scripts/run_tests.py --skip-integration

# Skip multiple suites
python ../scripts/run_tests.py --skip-agent --skip-gateway  # Only runs integration-test
```

### Manual Test Execution

```bash
cd services

# Run unit tests for a specific service
docker compose --profile test up agent-test
docker compose --profile test up gateway-test

# Run full system integration tests (requires gateway service running)
docker compose up -d gateway agent rabbitmq redis chroma
docker compose --profile test up integration-test
```

## Integration Tests

The `integration-test` service runs comprehensive end-to-end tests that:
- Make real HTTP API calls to the gateway service
- Verify RabbitMQ queue depths and message flow
- Check Redis task storage and status updates
- Verify worker counts and presence
- Test concurrent task submission
- Validate all API endpoints work correctly

These tests fully simulate real user interactions and verify the entire system is working correctly.

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

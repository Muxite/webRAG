# Retry Mechanisms and Failure Handling

## Overview

This document explains how retries work in the Euglena system and what happens when services take longer to start or go down during deployment.

## Retry Configuration

### Default Settings
- **Max Attempts**: 10 retries for all connectors (RabbitMQ, Redis, Chroma)
- **Base Delay**: 2 seconds (configurable via `DEFAULT_DELAY` env var)
- **Exponential Backoff**: Multiplier of 2.0 (doubles each attempt)
- **Max Delay**: 60 seconds (caps the exponential backoff)
- **Jitter**: 0.5 seconds (random variation to prevent thundering herd)

### Retry Timeline Example
With default settings (base_delay=2s, multiplier=2.0, max_delay=60s):
- Attempt 1: Immediate
- Attempt 2: ~2s delay
- Attempt 3: ~4s delay
- Attempt 4: ~8s delay
- Attempt 5: ~16s delay
- Attempt 6: ~32s delay
- Attempt 7: ~60s delay (capped)
- Attempt 8-10: ~60s delay each

**Total maximum retry time**: ~2 + 4 + 8 + 16 + 32 + 60 + 60 + 60 + 60 = **~302 seconds (~5 minutes)**

## Startup Retry Behavior

### Container Dependencies (ECS Task Definition)
All containers use `dependsOn` with `condition: START`:
- **Chroma**: No dependencies (starts first)
- **Redis**: No dependencies (starts first)
- **RabbitMQ**: No dependencies (starts first)
- **Agent**: Depends on Chroma, Redis, RabbitMQ (all START)
- **Gateway**: Depends on Chroma, Redis, RabbitMQ (all START)

**Note**: Dependencies use `START` condition, not `HEALTHY`. This means:
- Agent/Gateway start as soon as dependencies begin starting
- They don't wait for health checks to pass
- Application-level retries handle slow service startup

### Health Check Configuration
- **Chroma**: 300s start period, 90s interval, 6 retries, 15s timeout
- **Redis**: 120s start period, 90s interval, 6 retries, 15s timeout
- **RabbitMQ**: 300s start period, 90s interval, 6 retries, 15s timeout
- **Agent**: 300s start period, 90s interval, 6 retries, 15s timeout
- **Gateway**: 300s start period, 90s interval, 6 retries, 15s timeout

**Health checks are lenient** - they allow up to 5 minutes (300s) for services to become healthy before marking them as unhealthy.

### Application-Level Retries

#### RabbitMQ Connection
```python
# services/shared/connector_rabbitmq.py
retry = Retry(
    func=self._try_init_rabbitmq,
    max_attempts=10,
    base_delay=self.config.default_delay,  # 2s
    name="RabbitMQInit",
    jitter=self.config.jitter_seconds,  # 0.5s
)
```

**Behavior**:
- Retries connection up to 10 times with exponential backoff
- If all retries fail, logs error but doesn't crash
- `init_rabbitmq()` returns `False` on failure
- Agent/Gateway startup will fail if RabbitMQ doesn't connect

#### Redis Connection
```python
# services/shared/connector_redis.py
retry = Retry(
    func=self._try_init_redis,
    max_attempts=10,
    base_delay=self.config.default_delay,  # 2s
    name="RedisInit",
    jitter=self.config.jitter_seconds,  # 0.5s
)
```

**Behavior**:
- Retries connection up to 10 times with exponential backoff
- If all retries fail, logs error but doesn't crash
- `init_redis()` returns `False` on failure
- Operations gracefully handle Redis being unavailable (fail-open)

#### ChromaDB Connection
```python
# services/agent/app/connector_chroma.py
retry = Retry(
    func=self._try_init_chroma,
    max_attempts=10,
    base_delay=self.config.default_delay,  # 2s
    name="ChromaDBinit",
    jitter=self.config.jitter_seconds,  # 0.5s
)
```

**Behavior**:
- Retries connection up to 10 times with exponential backoff
- If all retries fail, logs error but doesn't crash
- `init_chroma()` returns `False` on failure
- Agent will fail to start if ChromaDB doesn't connect

## Runtime Failure Handling

### What Happens If a Service Goes Down During Runtime?

#### RabbitMQ Failure
**Detection**:
- `rabbitmq_ready` flag is set to `False` on connection failure
- Operations check `is_ready()` before use

**Reconnection**:
- **Automatic**: `init_rabbitmq()` is called before each operation
- **Retry Logic**: Uses same 10-attempt retry with exponential backoff
- **Operations Affected**:
  - `publish_message()`: Calls `init_rabbitmq()` - will retry connection
  - `get_channel()`: Calls `init_rabbitmq()` - will retry connection
  - `consume_queue()`: Calls `init_rabbitmq()` - will retry connection

**Behavior**:
- If RabbitMQ goes down, operations will automatically retry connection
- If reconnection succeeds, operations continue normally
- If reconnection fails after 10 attempts, operations raise `RuntimeError`

**Example Scenario**:
1. RabbitMQ container restarts (deployment, crash, etc.)
2. Agent/Gateway detect connection loss
3. Next operation (publish/consume) triggers `init_rabbitmq()`
4. Retry loop attempts reconnection with exponential backoff
5. If RabbitMQ comes back within ~5 minutes, connection succeeds
6. Operations resume normally

#### Redis Failure
**Detection**:
- `redis_ready` flag is set to `False` on connection failure
- Operations check `init_redis()` before use

**Reconnection**:
- **Automatic**: `init_redis()` is called before each operation via `get_client()`
- **Retry Logic**: Uses same 10-attempt retry with exponential backoff
- **Operations Affected**:
  - `get_json()`: Calls `get_client()` - will retry connection
  - `set_json()`: Calls `get_client()` - will retry connection
  - All storage operations: Call `get_client()` - will retry connection

**Behavior**:
- If Redis goes down, operations will automatically retry connection
- If reconnection succeeds, operations continue normally
- If reconnection fails, operations return `None` or `False` (fail-open)
- **Quota checks fail-open**: If Redis is unavailable, quota checks allow requests

**Example Scenario**:
1. Redis container restarts
2. Agent/Gateway detect connection loss on next operation
3. Operation triggers `init_redis()` via `get_client()`
4. Retry loop attempts reconnection with exponential backoff
5. If Redis comes back within ~5 minutes, connection succeeds
6. Operations resume normally

#### ChromaDB Failure
**Detection**:
- `chroma_api_ready` flag is set to `False` on connection failure
- Operations check `init_chroma()` before use

**Reconnection**:
- **Automatic**: `init_chroma()` is called before each operation
- **Retry Logic**: Uses same 10-attempt retry with exponential backoff
- **Operations Affected**:
  - `get_or_create_collection()`: Calls `init_chroma()` - will retry connection
  - `add_to_chroma()`: Checks `chroma_api_ready` - will retry if needed
  - `query_chroma()`: Checks `chroma_api_ready` - will retry if needed

**Behavior**:
- If ChromaDB goes down, operations will automatically retry connection
- If reconnection succeeds, operations continue normally
- If reconnection fails, operations return `None` or `False`

**Example Scenario**:
1. ChromaDB container restarts
2. Agent detects connection loss on next operation
3. Operation triggers `init_chroma()`
4. Retry loop attempts reconnection with exponential backoff
5. If ChromaDB comes back within ~5 minutes, connection succeeds
6. Operations resume normally

## Deployment Scenarios

### Scenario 1: Slow Service Startup
**What Happens**:
1. ECS starts containers in dependency order
2. Chroma/Redis/RabbitMQ start but take longer than expected
3. Agent/Gateway start immediately (START dependency, not HEALTHY)
4. Agent/Gateway attempt to connect to services
5. Connection retries handle slow startup (up to ~5 minutes)
6. Once services are ready, connections succeed
7. Services continue normally

**Result**: **Handled gracefully** - retries allow for slow startup

### Scenario 2: Service Crashes During Deployment
**What Happens**:
1. New task definition is deployed
2. Old task is stopped, new task starts
3. During transition, services may be temporarily unavailable
4. Agent/Gateway in new task retry connections
5. Once all services are up, connections succeed
6. Services continue normally

**Result**: **Handled gracefully** - retries handle temporary unavailability

### Scenario 3: Service Goes Down During Runtime
**What Happens**:
1. Service (RabbitMQ/Redis/Chroma) crashes or restarts
2. Active connections detect failure (on next operation)
3. Operation triggers reconnection via `init_*()` method
4. Retry loop attempts reconnection (up to ~5 minutes)
5. If service comes back, connection succeeds
6. Operations resume normally

**Result**: **Handled gracefully** - automatic reconnection with retries

### Scenario 4: Service Never Comes Back
**What Happens**:
1. Service fails and doesn't recover
2. Retry loop exhausts all 10 attempts (~5 minutes)
3. Connection fails permanently
4. **RabbitMQ**: Operations raise `RuntimeError` - service becomes unusable
5. **Redis**: Operations return `None`/`False` - service continues but with degraded functionality
6. **ChromaDB**: Operations return `None`/`False` - agent continues but can't store/query memory

**Result**: **Partial failure** - depends on which service failed

## Key Insights

### Strengths
1. **Exponential backoff** prevents overwhelming services during recovery
2. **Jitter** prevents thundering herd problems
3. **Automatic reconnection** handles transient failures
4. **Fail-open for Redis** allows service to continue even if Redis is down
5. **Lenient health checks** (300s start period) allow for slow startup

### Potential Issues
1. **No circuit breaker**: Services will keep retrying even if service is permanently down
2. **Long retry window**: ~5 minutes of retries may be too long for some use cases
3. **No health monitoring**: Services don't proactively check connection health
4. **RabbitMQ failure is fatal**: If RabbitMQ fails permanently, agent/gateway become unusable
5. **No connection pooling**: Each operation may trigger reconnection attempts

### Recommendations
1. **Add circuit breaker pattern**: Stop retrying after consecutive failures
2. **Reduce retry window**: Consider reducing max_attempts or max_delay for faster failure detection
3. **Add health monitoring**: Periodically check connection health proactively
4. **Add connection pooling**: Reuse connections instead of reconnecting on each operation
5. **Add metrics**: Track connection failures and retry attempts for observability

## Configuration

Retry behavior can be configured via environment variables:
- `DEFAULT_DELAY`: Base delay in seconds (default: 2)
- `JITTER_SECONDS`: Jitter in seconds (default: 0.5)
- `DEFAULT_TIMEOUT`: Timeout for operations (default: 5)

These are set in `services/.env` and loaded by `ConnectorConfig`.

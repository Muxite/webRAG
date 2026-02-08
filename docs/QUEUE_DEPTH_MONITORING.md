# Queue Depth Monitoring Architecture

## Decision: Gateway-Based Monitoring

**Decision**: Gateway service now handles queue depth monitoring instead of relying solely on the metrics service.

### Rationale

1. **Simpler Architecture**: Gateway already has a RabbitMQ connection, eliminating the need for a separate metrics container
2. **Reduced Failure Points**: Fewer containers means fewer potential failure modes
3. **Essential Service**: Gateway is always running (essential for the system), while metrics was optional
4. **Better Integration**: Queue depth is now available in the gateway health check endpoint
5. **Lower Resource Usage**: One less container to deploy and monitor

### Implementation

- **Gateway Service**: Periodically checks queue depths (default: every 5 seconds, configurable via `GATEWAY_QUEUE_DEPTH_INTERVAL`)
- **Health Check**: Gateway `/health` endpoint now includes `queue_depths` in the response
- **Metrics Service**: Still exists for CloudWatch publishing if needed, but gateway is the primary source

## Failure Modes

The `get_queue_depth()` method in `ConnectorRabbitMQ` can fail in several ways, each with a structured error code:

### Error Codes

1. **RABBITMQ_NOT_READY**
   - **Cause**: RabbitMQ connection not established
   - **Recovery**: Automatic reconnection via `init_rabbitmq()`
   - **Log Level**: WARNING

2. **CHANNEL_UNAVAILABLE**
   - **Cause**: Channel creation/retrieval failed
   - **Recovery**: Automatic channel recreation on next attempt
   - **Log Level**: WARNING

3. **CHANNEL_CLOSED**
   - **Cause**: Channel was closed before operation
   - **Recovery**: Sets `rabbitmq_ready = False`, triggers reconnection
   - **Log Level**: WARNING

4. **QUEUE_DECLARE_NONE**
   - **Cause**: Queue declaration returned None (unexpected)
   - **Recovery**: Retry on next interval
   - **Log Level**: WARNING

5. **NO_DECLARATION_RESULT**
   - **Cause**: Queue object missing `declaration_result` attribute
   - **Recovery**: Retry on next interval
   - **Log Level**: WARNING

6. **NO_MESSAGE_COUNT**
   - **Cause**: Declaration result missing `message_count` attribute
   - **Recovery**: Retry on next interval
   - **Log Level**: WARNING

7. **CHANNEL_CLOSED_DURING_OP**
   - **Cause**: Channel closed during queue operation
   - **Recovery**: Sets `rabbitmq_ready = False`, triggers reconnection
   - **Log Level**: WARNING

8. **CHANNEL_INVALID_STATE**
   - **Cause**: Channel in invalid state (e.g., during reconnection)
   - **Recovery**: Sets `rabbitmq_ready = False`, triggers reconnection
   - **Log Level**: WARNING

9. **UNKNOWN_ERROR**
   - **Cause**: Unexpected exception
   - **Recovery**: Retry on next interval
   - **Log Level**: WARNING (with full exception traceback)

## Logging Structure

All queue depth operations use structured logging with consistent fields:

### Success Log
```python
logger.debug(
    "Queue depth retrieved",
    extra={"queue": queue_name, "depth": depth, "error_code": None}
)
```

### Failure Log
```python
logger.warning(
    "Queue depth check failed: ERROR_CODE",
    extra={
        "queue": queue_name,
        "error_code": "ERROR_CODE",
        # Additional context fields...
    }
)
```

### Gateway Reporting Log
```python
logger.info(
    "Queue depth",
    extra={"queue": queue_name, "depth": depth}
)
```

## Configuration

### Gateway Queue Depth Monitoring

- **Environment Variable**: `GATEWAY_QUEUE_DEPTH_INTERVAL`
- **Default**: `5` seconds
- **Purpose**: Interval between queue depth checks
- **Queues Monitored**:
  - Primary input queue (`AGENT_INPUT_QUEUE`, default: `agent.mandates`)
  - Debug queue (`GATEWAY_DEBUG_QUEUE_NAME`, default: `gateway.debug`)

### Metrics Service (Optional)

- **Environment Variable**: `QUEUE_DEPTH_METRICS_INTERVAL`
- **Default**: `1` second
- **Purpose**: CloudWatch metric publishing interval
- **Note**: Metrics service can still run for CloudWatch integration, but gateway is the primary monitoring source

## Health Check Integration

The gateway `/health` endpoint now includes queue depth information:

```json
{
  "service": "gateway",
  "version": "0.1.0",
  "status": "healthy",
  "components": {
    "process": true,
    "rabbitmq": true,
    "redis": true
  },
  "queue_depths": {
    "agent.mandates": 0,
    "gateway.debug": 0
  }
}
```

If a queue depth is unavailable, it will be `null` in the response.

## Monitoring Best Practices

1. **Check Health Endpoint**: Monitor `/health` for queue depth values
2. **Watch for Error Codes**: Filter logs by `error_code` to identify specific failure modes
3. **Track Trends**: Queue depth should typically be 0-1 during normal operation
4. **Alert on Failures**: Set up alerts for repeated `RABBITMQ_NOT_READY` or `CHANNEL_CLOSED` errors
5. **Monitor Both Queues**: Both primary and debug queues should be monitored

## Migration Notes

- Gateway now reports queue depths every 5 seconds (configurable)
- Metrics service can be kept for CloudWatch publishing but is no longer required
- Health check endpoint provides real-time queue depth information
- All queue depth operations use structured logging with error codes

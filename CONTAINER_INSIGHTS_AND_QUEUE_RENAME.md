# Container Insights and Queue Rename

**Date**: 2026-02-05  
**Changes**: Enable Container Insights, rename debug_queue to gateway.debug

## Changes Made

### 1. Container Insights Enabled

**File**: `scripts/ecs_infrastructure.py`

- Added Container Insights to cluster creation
- Added Container Insights check and enablement for existing clusters
- Required for ECS task-level metrics in CloudWatch

**Impact**: Task-level metrics will now be available in CloudWatch without manual configuration.

### 2. Queue Renamed: `debug_queue` → `gateway.debug`

**Rationale**: 
- More descriptive name indicating it's a gateway-specific debug queue
- Uses dot notation consistent with other queues (`agent.mandates`, `agent.status`)
- Set via environment variable for configuration flexibility

**Files Modified**:

#### Configuration
- `services/shared/connector_config.py`: Added `gateway_debug_queue_name` property (default: `"gateway.debug"`)

#### Gateway Service
- `services/gateway/app/gateway_service.py`: 
  - Uses `self.config.gateway_debug_queue_name` instead of direct env var
  - Changed env var from `DEBUG_QUEUE_NAME` to `GATEWAY_DEBUG_QUEUE_NAME`
  - Changed env var from `DEBUG_QUEUE_PHRASE` to `GATEWAY_DEBUG_QUEUE_PHRASE`

#### Metrics Service
- `services/metrics/app/main.py`: Uses `cfg.gateway_debug_queue_name` for code reuse
- `autoscale/services/metrics/app/main.py`: Uses `cfg.gateway_debug_queue_name` for code reuse

#### Task Definition Scripts
- `scripts/build-task-definition.py`: 
  - Changed `DEBUG_QUEUE_NAME` → `GATEWAY_DEBUG_QUEUE_NAME` (default: `"gateway.debug"`)
  - Changed `DEBUG_QUEUE_PHRASE` → `GATEWAY_DEBUG_QUEUE_PHRASE`
- `autoscale/scripts/build-task-definition.py`: Same changes

## Environment Variables

### New/Updated Variables

**Gateway Container**:
- `GATEWAY_DEBUG_QUEUE_NAME`: Queue name for gateway debug messages (default: `"gateway.debug"`)
- `GATEWAY_DEBUG_QUEUE_PHRASE`: Phrase to detect debug messages (default: `"debugdebugdebug"`)

**Metrics Container**:
- Uses `GATEWAY_DEBUG_QUEUE_NAME` from ConnectorConfig (shared code)

### Setting in .env

Add to `services/.env`:
```bash
GATEWAY_DEBUG_QUEUE_NAME=gateway.debug
GATEWAY_DEBUG_QUEUE_PHRASE=debugdebugdebug
```

## Code Reuse

The queue name is now centralized in `ConnectorConfig`:
- Single source of truth: `connector_config.py`
- Gateway service uses: `self.config.gateway_debug_queue_name`
- Metrics service uses: `cfg.gateway_debug_queue_name`
- Both services share the same configuration object

## Benefits

1. **Container Insights**: Automatic task-level metrics in CloudWatch
2. **Better naming**: `gateway.debug` is more descriptive than `debug_queue`
3. **Code reuse**: Single configuration source for queue name
4. **Environment-based**: Can be configured via .env file
5. **Consistent**: Uses dot notation like other queues

## Migration Notes

- Old queue name `debug_queue` will be automatically created if referenced
- New code will use `gateway.debug` by default
- Existing messages in `debug_queue` will remain until consumed
- No data migration needed - new messages will go to `gateway.debug`

## Next Steps

1. **Deploy**: `cd services && python ../scripts/deploy-autoscale.py`
2. **Verify Container Insights**: Check CloudWatch for task-level metrics
3. **Test queue**: Submit debug message and verify it goes to `gateway.debug`
4. **Monitor**: Check metrics logs for `gateway.debug` queue depth

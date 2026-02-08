# Stability Fixes Summary

**Date**: 2026-02-05  
**Issue**: Deployment instability, RabbitMQ shutdowns, debug_queue errors

## Root Causes Identified

1. **debug_queue doesn't exist**: Metrics service tries to check queue depth for `debug_queue` but it doesn't exist, causing channel errors
2. **Hardcoded values**: Gateway service uses hardcoded `"debug_queue"` and `"debugdebugdebug"` instead of environment variables
3. **get_queue_depth fails on non-existent queues**: Uses `passive=True` which fails if queue doesn't exist
4. **Deployment instability**: Force deployment always triggered, aggressive task stopping

## Fixes Applied

### 1. Fixed `get_queue_depth` to Handle Non-Existent Queues
**File**: `services/shared/connector_rabbitmq.py`

- Changed from `passive=True` (fails if queue doesn't exist) to try-catch approach
- If queue doesn't exist, declare it with `durable=True`
- Handles channel errors gracefully

**Impact**: Metrics service can now check `debug_queue` even if it doesn't exist yet - it will be created automatically.

### 2. Replaced Hardcoded Values with Environment Variables
**Files**: 
- `services/gateway/app/gateway_service.py`
- `scripts/build-task-definition.py`
- `autoscale/scripts/build-task-definition.py`

**Changes**:
- Gateway now uses `DEBUG_QUEUE_NAME` and `DEBUG_QUEUE_PHRASE` environment variables
- Task definitions set these environment variables with defaults
- Gateway environment includes `DEBUG_QUEUE_NAME` and `DEBUG_QUEUE_PHRASE`

**Impact**: Configuration is now centralized and can be changed via environment variables.

### 3. Fixed Deployment Stability Issues
**Files**: 
- `scripts/ecs_infrastructure.py`
- `scripts/deploy_ecs.py`
- `scripts/deploy-autoscale.py`

**Changes**:
- Only force deployment when task definition actually changes
- Removed aggressive `stop_old_tasks()` that was causing RabbitMQ shutdowns
- Added default 60-second wait for service startup

**Impact**: Fewer unnecessary deployments, no forced task stops, better startup visibility.

### 4. Fixed Gateway Health Check Period
**File**: `autoscale/scripts/build-task-definition.py`

- Changed gateway `start_period` from 600s to 300s (AWS ECS maximum)
- Prevents task definition registration failures

**Impact**: Compliant task definitions that won't be rejected by ECS.

## Environment Variables Added

### Gateway Container
- `DEBUG_QUEUE_NAME`: Queue name for debug messages (default: `"debug_queue"`)
- `DEBUG_QUEUE_PHRASE`: Phrase to detect debug messages (default: `"debugdebugdebug"`)

### Metrics Container
- `DEBUG_QUEUE_NAME`: Queue name to monitor (default: `"debug_queue"`)
- `QUEUE_NAME`: Primary queue to monitor (default: `"agent.mandates"`)

## Expected Behavior After Fixes

1. **No more debug_queue errors**: Queue will be created automatically when first checked
2. **Configurable debug queue**: Can change queue name/phrase via environment variables
3. **Stable deployments**: Only deploys when task definitions actually change
4. **No RabbitMQ shutdowns**: Tasks replaced gracefully without forced stops
5. **Proper health checks**: Gateway health check period within AWS limits

## Testing Recommendations

1. **Deploy and verify**:
   ```bash
   cd services
   python ../scripts/deploy-autoscale.py
   ```

2. **Check metrics logs**: Should see "Queue Backlog: queue=debug_queue, waiting=0" without errors

3. **Submit debug message**: Should route to debug_queue and accumulate there

4. **Monitor RabbitMQ**: Should not see SIGTERM signals during deployment

## Files Modified

- `services/shared/connector_rabbitmq.py`: Fixed queue depth check
- `services/gateway/app/gateway_service.py`: Use environment variables
- `scripts/build-task-definition.py`: Add DEBUG_QUEUE_NAME/DEBUG_QUEUE_PHRASE to gateway
- `autoscale/scripts/build-task-definition.py`: Add DEBUG_QUEUE_NAME/DEBUG_QUEUE_PHRASE, fix gateway start_period
- `scripts/ecs_infrastructure.py`: Conditional force deployment
- `scripts/deploy_ecs.py`: Removed aggressive task stopping
- `scripts/deploy-autoscale.py`: Added default wait

## Next Steps

1. Deploy with `deploy-autoscale.py`
2. Monitor logs for debug_queue errors (should be gone)
3. Verify metrics show both queues correctly
4. Test debug message routing
5. Verify deployment stability (no unnecessary redeployments)

# Resource Increase for Gateway Task

**Date**: 2026-02-05  
**Issue**: RabbitMQ container killed due to OutOfMemoryError (Exit code: 137)

## Problem

RabbitMQ container in the gateway task was being killed due to memory exhaustion:
```
Task stopped at: 2026-02-05T23:20:39.713Z
Essential container in task exited
1 essential container exited
[rabbitmq] Exit code: 137, reason: "OutOfMemoryError: Container killed due to memory usage".
```

## Solution

Increased gateway task resources:
- **CPU**: 512 → **1024** (0.5 vCPU → 1.0 vCPU)
- **Memory**: 1024 MB → **2048 MB** (1 GB → 2 GB)

### Changes Made

**Files Modified**:
- `scripts/build-task-definition.py`: Gateway task memory: 1024 → 2048 MB
- `autoscale/scripts/build-task-definition.py`: Gateway task memory: 1024 → 2048 MB

**Configuration**:
- **CPU**: 1024 (1.0 vCPU) - doubled from 512
- **Memory**: 2048 MB (2 GB) - doubled from 1024 MB
- **Containers**: 5 (chroma, redis, rabbitmq, gateway, metrics)

### Valid Fargate Configuration

For 1024 CPU, valid memory options are:
- 2048 MB (selected)
- 3072 MB
- 4096 MB
- 5120 MB
- 6144 MB
- 7168 MB
- 8192 MB

## Impact

- **More CPU and memory available** for all containers in the gateway task
- **RabbitMQ should no longer be killed** due to memory pressure
- **Better performance** with doubled CPU resources
- **Better stability** for all gateway services
- **Higher cost** (CPU and memory are billed, but should prevent task failures and improve performance)

## Next Steps

1. **Deploy the updated configuration**:
   ```bash
   cd services
   python ../scripts/deploy-autoscale.py
   ```

2. **Monitor deployment**:
   - Check that new task definition is registered
   - Verify services start successfully
   - Monitor RabbitMQ logs for memory issues

3. **Audit after deployment**:
   - Use `audit-aws-changes.py` to verify stability
   - Check task definition revisions
   - Monitor for any memory-related errors

## Notes

- CPU and memory are shared across all 5 containers in the gateway task
- If memory issues persist, consider:
  - Increasing memory to 3072 MB, 4096 MB, or higher (up to 8192 MB with 1024 CPU)
  - Adding memory limits to individual containers (if needed)

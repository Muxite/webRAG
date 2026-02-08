# RabbitMQ Task Failure Investigation

**Date**: 2026-02-05  
**Task ARN**: `arn:aws:ecs:us-east-2:848960888155:task/euglena-cluster/cbb8903ef5264c9b9621ff182ab9c6a2`  
**Issue**: Task failed ELB health checks, RabbitMQ container issues suspected

## Analysis Framework

Created `scripts/comprehensive-audit.py` that combines:
- `audit-aws-changes.py` - AWS change tracking
- `analyze-audit.py` - Breaking point analysis  
- New task-level investigation capabilities

## Key Investigation Areas

### 1. Task-Level Analysis
- Container exit codes and reasons
- Memory/CPU allocation vs usage
- Health check configurations
- Stop codes and reasons

### 2. RabbitMQ-Specific Issues
- Exit code 137 = OOM (Out of Memory) kill
- Memory allocation per container
- Health check timing
- Container dependencies

### 3. Resource Analysis
- Task definition CPU/memory changes
- Container count changes
- Memory per container calculations

## Recent Changes That Could Affect RabbitMQ

### Memory Increase (r68)
- **CPU**: 512 → 1024 (doubled)
- **Memory**: 1024 → 2048 MB (doubled)
- **Impact**: Should help, but need to verify if sufficient

### Container Insights
- Enabled on cluster
- Should provide better visibility into resource usage

### Queue Rename
- `debug_queue` → `gateway.debug`
- Should not affect RabbitMQ stability

## Potential Root Causes

### 1. Memory Pressure
- **Symptom**: Exit code 137 (OOM kill)
- **Analysis**: 2048 MB shared across 5 containers = ~409 MB per container
- **RabbitMQ needs**: Typically 256-512 MB minimum
- **Other containers**: Chroma, Redis, Gateway, Metrics also need memory
- **Recommendation**: May need 3072-4096 MB total

### 2. Health Check Timing
- **RabbitMQ health check**: start_period=300s, interval=90s, timeout=15s
- **Gateway health check**: start_period=300s, interval=90s, timeout=15s
- **Issue**: If RabbitMQ fails health check, gateway can't start properly
- **Recommendation**: Verify RabbitMQ health check is appropriate

### 3. Container Dependencies
- **Gateway depends on**: Redis, RabbitMQ
- **Metrics depends on**: RabbitMQ
- **Issue**: If RabbitMQ fails, dependent containers fail
- **Recommendation**: Ensure RabbitMQ is stable before dependent containers start

### 4. Network/Connectivity
- **ELB health checks failing**: Gateway container can't respond
- **Possible cause**: RabbitMQ not ready, so gateway can't initialize
- **Recommendation**: Check gateway logs for RabbitMQ connection errors

## Investigation Commands

### Run Comprehensive Audit
```bash
python scripts/comprehensive-audit.py \
  --task-arn "arn:aws:ecs:us-east-2:848960888155:task/euglena-cluster/cbb8903ef5264c9b9621ff182ab9c6a2" \
  --target-group-arn "arn:aws:elasticloadbalancing:us-east-2:848960888155:targetgroup/euglena-tg/ea4bbe2f98578c2a" \
  --days 3 \
  --output rabbitmq-investigation.json
```

### Check Recent Failures
```bash
python scripts/comprehensive-audit.py \
  --days 1 \
  --output recent-failures.json
```

## Next Steps

1. **Run audit** (after AWS credentials refresh)
2. **Analyze task details** for container exit codes
3. **Check memory allocation** - may need to increase to 3072-4096 MB
4. **Review RabbitMQ health check** configuration
5. **Check CloudWatch logs** for RabbitMQ container errors
6. **Verify container dependencies** are correct

## Recommendations

### Immediate
1. **Increase memory to 3072 MB** if 2048 MB is insufficient
2. **Check RabbitMQ logs** for specific error messages
3. **Verify health check timing** is appropriate for RabbitMQ startup

### Configuration
1. **Add memory limits** to individual containers if needed
2. **Adjust health check grace period** if RabbitMQ needs more time
3. **Monitor Container Insights** metrics for actual memory usage

### Long-term
1. **Set up CloudWatch alarms** for OOM kills
2. **Add memory usage metrics** to track per-container usage
3. **Document memory requirements** for each container

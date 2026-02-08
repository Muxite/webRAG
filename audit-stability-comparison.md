# Stability Audit Comparison: Current vs Stable

**Audit Date**: 2026-02-05  
**Stable Reference**: Gateway r56, Agent r55 (2026-02-04)  
**Current Deployment**: Gateway r64, Agent r63 (2026-02-05)

## Resource Configuration Comparison

### Gateway Service
| Metric | Stable (r56) | Current (r64) | Status |
|--------|-------------|---------------|--------|
| CPU | 512 | 512 | ✅ MATCHES |
| Memory | 1024 MB | 1024 MB | ✅ MATCHES |
| Containers | 5 | 5 | ✅ MATCHES |
| Container Names | chroma, redis, rabbitmq, gateway, metrics | chroma, redis, rabbitmq, gateway, metrics | ✅ MATCHES |

### Agent Service
| Metric | Stable (r55) | Current (r63) | Status |
|--------|-------------|---------------|--------|
| CPU | 256 | 256 | ✅ MATCHES |
| Memory | 512 MB | 512 MB | ✅ MATCHES |
| Containers | 1 | 1 | ✅ MATCHES |

## Deployment Activity Analysis

### Task Definition Revisions (Last 3 Days)
- **Gateway**: 10 revisions (r55 → r64) - **8 revisions since stable r56**
- **Agent**: 10 revisions (r54 → r63) - **8 revisions since stable r55**
- **Frequency**: Multiple revisions per day (r64 at 04:41, r63 at 04:37, r62 at 03:06)

### Key Findings

#### ✅ **Good Signs**
1. **Resource configuration unchanged**: All revisions maintain stable CPU/memory/container counts
2. **No container count changes**: Gateway consistently has 5 containers, agent has 1
3. **Container names consistent**: All gateway revisions include metrics container

#### ⚠️ **Warning Signs**
1. **High revision frequency**: 8 gateway + 8 agent revisions in ~24 hours suggests instability-driven redeployments
2. **All revisions orphaned**: No git commits nearby any task definition registrations
   - Suggests deployments without code changes (config-only or image rebuilds)
   - Could indicate repeated attempts to fix instability
3. **ECR push timing**: 
   - Metrics image last pushed: 2026-02-04 02:29 (before new code changes)
   - Agent image last pushed: 2026-02-03 17:01
   - **New metrics code (dual queue, 1-second interval) not yet deployed**

## Current Instability Issues

### 1. RabbitMQ Container Shutdowns
**Evidence from logs:**
- SIGTERM received at 2026-02-05 12:47:17
- "Virtual host '/' is stopping"
- Message stores stopping
- This causes gateway health check failures and ELB target unhealthiness

**Root Cause Analysis:**
- RabbitMQ health check: start_period=300, interval=90, retries=6, timeout=15
- Container may be hitting resource limits or health check timing issues
- Could be related to frequent task definition updates causing container restarts

### 2. Metrics Service Not Updated
**Evidence:**
- Logs show old format: `[INFO] [MetricsService] Queue depth: agent.mandates=0`
- Should show: `[INFO] [MetricsService] Queue Backlog: queue=agent.mandates, waiting=0`
- No mention of `debug_queue` in logs
- Metrics image last pushed before code changes (2026-02-04)

**Impact:**
- Cannot verify queue backlog monitoring is working
- Cannot test debug_queue accumulation

## Recommendations

### Immediate Actions

1. **Rebuild and Deploy Metrics Image**
   ```bash
   cd services
   python ../scripts/deploy-autoscale.py
   ```
   - This will build new metrics image with dual queue monitoring
   - Will enable "Queue Backlog" logging for both queues

2. **Investigate RabbitMQ Stability**
   - Check ECS service events for RabbitMQ container failures
   - Verify RabbitMQ health check timing is appropriate
   - Consider if resource constraints are causing shutdowns
   - Check if frequent task definition updates are triggering restarts

3. **Reduce Deployment Frequency**
   - Current: 8 revisions in 24 hours
   - Target: Only deploy when code/config actually changes
   - Use `--skip-ecr` flag when only task definition config changes

### Stability Verification

After deploying new metrics image, verify:
1. Metrics logs show "Queue Backlog" format for both queues
2. RabbitMQ container stays running (no SIGTERM signals)
3. Gateway health checks pass consistently
4. ALB target remains healthy

### Rollback Plan

If instability continues:
1. Rollback to stable task definitions:
   ```bash
   aws ecs update-service --cluster euglena-cluster --service euglena-gateway --task-definition euglena-gateway:56 --region us-east-2
   aws ecs update-service --cluster euglena-cluster --service euglena-agent --task-definition euglena-agent:55 --region us-east-2
   ```
2. Wait for services to stabilize
3. Investigate what changed between r56/r55 and r64/r63

## Summary

**Configuration**: ✅ Matches stable (resources unchanged)  
**Deployment Frequency**: ⚠️ High (8 revisions in 24h - instability indicator)  
**RabbitMQ**: ❌ Unstable (SIGTERM shutdowns)  
**Metrics**: ❌ Not updated (old code still running)  

**Primary Issue**: RabbitMQ container instability causing gateway health check failures. Configuration matches stable, but frequent redeployments and RabbitMQ shutdowns indicate underlying stability problems.

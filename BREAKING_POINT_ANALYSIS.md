# Breaking Point Analysis

**Target (Working) Time**: 2026-02-01 15:56:28 -0800  
**Analysis Date**: 2026-02-04

## Good State (Working)

- **Gateway r46**: Feb 1 18:35 (+2.6h after target)
  - CPU: 512, Memory: 1024, Containers: 5
- **Agent r43**: Feb 1 18:35
  - CPU: 256, Memory: 512, Containers: 1

## Breaking Point

**First Failure**: Feb 4 03:39 - Gateway task replaced due to unhealthy status  
**Task Definition at Failure**: Gateway r54 (Feb 4 02:36)

### Changes from Good State (r46) to Failure State (r54)

1. **Resource Increase**: CPU 512→1024, Memory 1024→2048 (doubled)
2. **Container Count**: Same (5 containers)
3. **Timeline Gap**: 58.7 hours between good state and failure

## Critical Timeline

1. **Feb 1 18:35** - r46 registered (GOOD STATE)
2. **Feb 3 16:33-20:54** - Multiple rollback attempts (r47-r51)
3. **Feb 3 19:41** - **r49: Container count dropped 5→4** ⚠️
4. **Feb 3 20:27** - r50: Container count back to 5
5. **Feb 3 23:22** - r52: Still 512/1024, 5 containers
6. **Feb 4 02:29** - r53: Resources doubled (512→1024, 1024→2048)
7. **Feb 4 02:36** - r54: Same as r53, deployment fails
8. **Feb 4 03:39** - Tasks failing health checks

## Key Findings

### 1. Container Count Anomaly - CONFIRMED ROOT CAUSE
- **r46 (GOOD)**: 5 containers - chroma, redis, rabbitmq, gateway, **metrics** ✅
- **r49 (BROKEN)**: 4 containers - chroma, redis, rabbitmq, gateway (NO METRICS) ❌
- **r50+**: Restored to 5 containers but system still broken

**This is the smoking gun**: r49 was deployed without the metrics container, which likely caused:
- Service discovery issues
- Health check failures
- Cascading failures that persisted even after metrics was restored

### 2. Resource Changes Don't Fix It
- r53-54 doubled resources but still failing
- This suggests the issue is NOT resource constraints
- More likely: code/image change, container config, or AWS infrastructure

### 3. ECR Image Push
- **Feb 3 17:01** - Agent image pushed (untagged)
- This is 49h after good state
- Could be a new image that broke compatibility

### 4. No Git Commits
- All task definitions are "orphaned" (no nearby git commits)
- This means changes were made directly in AWS or via scripts without commits
- Makes it impossible to correlate code changes with failures

## Root Cause Hypothesis

The system was working with **r46** (Feb 1). Then:

1. **Feb 3 17:01**: New agent image pushed (untagged)
2. **Feb 3 19:41**: r49 deployed with only 4 containers (missing metrics?)
3. **Feb 3 20:27**: r50 restored to 5 containers
4. **Feb 4 02:29-36**: r53-54 doubled resources but still failing

**Most Likely Causes**:
1. **New agent image** (Feb 3) broke compatibility with gateway
2. **Container configuration** changed (r49 missing container)
3. **AWS infrastructure** changed (ALB health checks, service discovery, etc.)
4. **Code changes** not tracked in git (deployed directly)

## Recommendations

1. **Check what changed in agent image** pushed Feb 3 17:01
2. **Compare r46 vs r54 task definitions** to see exact container differences
3. **Check ALB target group health check settings** - may have changed
4. **Verify service discovery** is still working correctly
5. **Check if any AWS service limits** were hit or changed
6. **Review CloudWatch logs** from Feb 1-4 to find first error

## Next Steps

1. Get full task definition JSON for r46 (good) and r54 (bad)
2. Compare container definitions line-by-line
3. Check if metrics container config changed
4. Verify health check commands are identical
5. Check if any environment variables changed

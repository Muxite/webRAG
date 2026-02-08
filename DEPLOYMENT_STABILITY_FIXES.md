# Deployment Stability Fixes

**Date**: 2026-02-05  
**Issue**: Deployment causing instability (RabbitMQ shutdowns, frequent redeployments)

## Changes Made

### 1. Conditional Force Deployment (`scripts/ecs_infrastructure.py`)

**Problem**: `forceNewDeployment: True` was always set, causing new deployments even when task definitions hadn't changed. This led to unnecessary container restarts and instability.

**Fix**: Only force deployment when task definition actually changes:
- Check current task definition ARN vs latest task definition ARN
- Only set `forceNewDeployment: True` if they differ
- Log when skipping force deployment due to unchanged task definition

**Impact**: Reduces unnecessary deployments and container restarts, improving stability.

### 2. Removed Aggressive Task Stopping (`scripts/deploy_ecs.py`)

**Problem**: `stop_old_tasks()` was called before every deployment, stopping all running tasks. This caused RabbitMQ to receive SIGTERM signals and shut down, leading to gateway health check failures.

**Fix**: Removed `stop_old_tasks()` call from `ensure_exact_service_config()`. ECS will handle task replacement gracefully during rolling deployments.

**Impact**: Prevents RabbitMQ shutdowns and maintains service continuity during deployments.

### 3. Default Wait for Service Startup (`scripts/deploy-autoscale.py`)

**Problem**: Deployment completed immediately without waiting for services to start, making it difficult to verify stability.

**Fix**: Added default 60-second wait after service updates (even without `--wait` flag) to allow initial startup. Full stability check still requires `--wait` flag.

**Impact**: Better visibility into deployment status and allows services to start before deployment script exits.

### 4. Metrics Service Deployment

**Status**: Already included in deployment (`ecr_services = ["gateway", "agent", "metrics"]`)

**Verification**: Metrics service is built and pushed to ECR as part of normal deployment flow.

## Expected Behavior After Fixes

1. **Fewer Unnecessary Deployments**: Only deploys when task definition actually changes
2. **No RabbitMQ Shutdowns**: Tasks are replaced gracefully without forced stops
3. **Better Startup Visibility**: Default wait allows services to start before script exits
4. **Metrics Always Updated**: Metrics service is included in every deployment

## Testing Recommendations

1. **Deploy with unchanged task definition**: Should skip force deployment
2. **Deploy with changed task definition**: Should force deployment and update services
3. **Monitor RabbitMQ logs**: Should not see SIGTERM signals during deployment
4. **Check metrics logs**: Should show "Queue Backlog" format after deployment

## Rollback Plan

If issues persist:
1. Revert `scripts/ecs_infrastructure.py` line 191: change `"forceNewDeployment": task_def_changed` back to `"forceNewDeployment": True`
2. Revert `scripts/deploy_ecs.py`: add back `stop_old_tasks()` call
3. Revert `scripts/deploy-autoscale.py`: remove default wait

## Related Files

- `scripts/ecs_infrastructure.py`: Service update logic
- `scripts/deploy_ecs.py`: ECS service management
- `scripts/deploy-autoscale.py`: Main deployment script
- `audit-stability-comparison.md`: Audit results showing instability issues

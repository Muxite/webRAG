# Stable Deployment Configuration

**Last Verified Stable**: 2026-02-05  
**Stable Task Definition**: Gateway r68+ (1024 CPU / 2048 MB), Agent r55+  
**Based On**: Gateway r56 (2026-02-04) - Updated with increased resources to prevent RabbitMQ OOM

## Critical Fix Applied

**VPC DNS Configuration**: Enabled DNS resolution and DNS support in VPC `vpc-02cc22c217f55e04b` to allow agent to resolve service discovery DNS names. This is REQUIRED for autoscale mode to function.

## Stable Resource Configuration

### Gateway Service
- **CPU**: 1024 (1.0 vCPU)
- **Memory**: 2048 MB (2 GB)
- **Containers**: 5 (chroma, redis, rabbitmq, gateway, metrics)
- **Memory per Container**: ~409 MB average (sufficient for RabbitMQ)

### Agent Service
- **CPU**: 256 (0.25 vCPU)
- **Memory**: 512 MB (0.5 GB)
- **Containers**: 1 (agent)

## Critical Configuration Values

### Health Checks

#### Chroma Container
- **startPeriod**: 300 seconds (AWS ECS maximum)
- **interval**: 60 seconds
- **retries**: 6
- **timeout**: 30 seconds
- **Command**: `curl -f http://localhost:8000/api/v1/heartbeat || (echo 'Chroma health check failed' && exit 1)`

**IMPORTANT**: AWS ECS enforces a maximum startPeriod of 300 seconds. Values above this will cause task definition registration to fail.

#### Gateway Container
- **startPeriod**: 300 seconds
- **interval**: 90 seconds
- **retries**: 6
- **timeout**: 15 seconds
- **Command**: `curl -f http://localhost:8080/health || exit 1`

#### Metrics Container
- **startPeriod**: 60 seconds
- **interval**: 30 seconds
- **retries**: 3
- **timeout**: 10 seconds
- **Command**: `curl -f http://localhost:8082/health || exit 1`
- **essential**: false (non-essential container)

### Container Dependencies

Gateway task definition container order:
1. **chroma** - No dependencies
2. **redis** - No dependencies
3. **rabbitmq** - No dependencies
4. **gateway** - Depends on: redis, rabbitmq
5. **metrics** - Depends on: rabbitmq

## Known Issues and Fixes

### Issue 1: Missing Metrics Container (r49)
- **Symptom**: Task definition had only 4 containers instead of 5
- **Root Cause**: Manual task definition edit or deployment script bug
- **Fix**: Ensure `build_gateway_task_definition()` always includes metrics container
- **Location**: `scripts/build-task-definition.py` line 352: `containers = [chroma, redis, rabbitmq, gateway, metrics]`

### Issue 2: Resource Over-allocation (r53-r54)
- **Symptom**: Gateway resources doubled (1024 CPU, 2048 memory) but still failing
- **Root Cause**: Attempted to fix stability by increasing resources
- **Fix**: Reverted to stable r46 configuration (512 CPU, 1024 memory)
- **Location**: `scripts/build-task-definition.py` lines 393-394

### Issue 4: RabbitMQ OOM Kills (r68+)
- **Symptom**: RabbitMQ container killed with exit code 137 (Out of Memory)
- **Root Cause**: 1024 MB shared across 5 containers insufficient for RabbitMQ
- **Fix**: Increased to 1024 CPU / 2048 MB (provides ~409 MB per container)
- **Location**: `scripts/build-task-definition.py` and `autoscale/scripts/build-task-definition.py`

### Issue 3: Chroma Health Check startPeriod Too High
- **Symptom**: Task definition registration failed with "startPeriod must be less than or equal to 300"
- **Root Cause**: startPeriod set to 600 seconds (exceeds AWS ECS limit)
- **Fix**: Changed to 300 seconds (AWS maximum)
- **Location**: `scripts/build-task-definition.py` lines 239, 510

## How to Modify Configuration

### Changing Gateway Resources

Edit `scripts/build-task-definition.py`:

```python
# Line 393-394 in build_gateway_task_definition()
return {
    # ... other fields ...
    "cpu": "1024",     # Change this (in CPU units: 256 = 0.25 vCPU, 512 = 0.5 vCPU, 1024 = 1 vCPU)
    "memory": "2048"   # Change this (in MB)
}
```

**Valid CPU/Memory combinations for Fargate**:
- 256 CPU: 512, 1024, 2048 MB
- 512 CPU: 1024, 2048, 3072, 4096 MB
- 1024 CPU: 2048, 3072, 4096, 5120, 6144, 7168, 8192 MB

**Current Stable Configuration (r68+)**:
- Gateway: 1024 CPU / 2048 MB (valid combination)
- Agent: 256 CPU / 512 MB (valid combination)
- Memory per container (Gateway): ~409 MB average (sufficient for RabbitMQ)

### Changing Health Check Parameters

Edit `scripts/build-task-definition.py`:

```python
# Example: Chroma health check (line 239)
"healthCheck": _make_health_check(
    chroma_health_command,
    start_period=300,  # Maximum 300 seconds (AWS limit)
    interval=60,       # Check interval in seconds
    retries=6,         # Number of consecutive failures before unhealthy
    timeout=30         # Timeout per check in seconds
)
```

**Constraints**:
- `startPeriod`: Maximum 300 seconds (AWS ECS limit)
- `interval`: Minimum 10 seconds
- `timeout`: Must be less than `interval`
- `retries`: Typically 3-6

### Adding/Removing Containers

Edit `scripts/build-task-definition.py`:

1. Define container in `build_gateway_task_definition()` function
2. Add to containers list (line 352):
   ```python
   containers = [chroma, redis, rabbitmq, gateway, metrics, new_container]
   ```
3. Ensure container image exists in ECR or is publicly available
4. Update task CPU/memory if needed to accommodate new container

### Modifying Container Dependencies

Edit container definition in `scripts/build-task-definition.py`:

```python
container = {
    # ... other fields ...
    "dependsOn": [
        {"condition": "START", "containerName": "redis"},
        {"condition": "START", "containerName": "rabbitmq"}
    ]
}
```

**Dependency conditions**:
- `START`: Container must start successfully before this container starts
- `SUCCESS`: Container must exit successfully before this container starts
- `COMPLETE`: Container must exit (success or failure) before this container starts
- `HEALTHY`: Container must pass health check before this container starts

## Deployment Process

### Standard Deployment
```bash
cd services
python ../scripts/deploy-autoscale.py
```

### Skip ECR Push (Use Existing Images)
```bash
cd services
python ../scripts/deploy-autoscale.py --skip-ecr
```

### Wait for Stability
```bash
cd services
python ../scripts/deploy-autoscale.py --wait
```

### Verify Deployment
```bash
cd services
python ../scripts/check-autoscale.py
```

## Verification Checklist

After deployment, verify:

- [ ] Gateway service: 1/1 running
- [ ] Agent service: 1/1 running
- [ ] ALB target: 1 healthy
- [ ] Container health: At least 4/5 healthy in gateway (metrics is non-essential)
- [ ] Service discovery: 1 instance registered
- [ ] No pending tasks
- [ ] No deployment failures in ECS events

## Rollback Procedure

If deployment fails:

1. **Identify last stable revision**:
   ```bash
   aws ecs list-task-definitions --family-prefix euglena-gateway --sort DESC --max-items 10
   ```

2. **Update service to use stable revision**:
   ```bash
   aws ecs update-service --cluster euglena-cluster --service euglena-gateway --task-definition euglena-gateway:56 --region us-east-2
   ```

3. **Wait for service to stabilize**:
   ```bash
   aws ecs wait services-stable --cluster euglena-cluster --services euglena-gateway --region us-east-2
   ```

## Key Files

- **Task Definition Builder**: `scripts/build-task-definition.py`, `autoscale/scripts/build-task-definition.py`
- **Deployment Script**: `scripts/deploy-autoscale.py`
- **Health Check**: `scripts/check-autoscale.py`
- **Comprehensive Audit**: `scripts/comprehensive-audit.py` (unified audit tool)
- **Config Capture**: `scripts/capture-stable-config.py` (capture config from AWS)
- **Config Documentation**: `scripts/generate-stable-config-doc.py` (generate markdown)

## Configuration Capture

To capture the current stable configuration from AWS:

```bash
# 1. Capture from AWS
python scripts/capture-stable-config.py \
  --services euglena-gateway euglena-agent \
  --target-group-arn "arn:aws:elasticloadbalancing:us-east-2:848960888155:targetgroup/euglena-tg/ea4bbe2f98578c2a" \
  --output stable-config.json

# 2. Generate documentation
python scripts/generate-stable-config-doc.py stable-config.json STABLE_CONFIG.md
```

## Known Issues

### Issue: Agent Cannot Connect to Gateway Services (Autoscale Mode)

**Symptom**: Agent health shows `rabbitmq=FAIL, redis=FAIL, chroma=FAIL` even though URLs are correctly set to `euglena-gateway.euglena.local`.

**Root Cause**: VPC DNS resolution and DNS support were disabled, preventing the agent from resolving service discovery DNS names.

**Required Configuration**:
1. **VPC DNS**: MUST be enabled for service discovery to work
   - `EnableDnsResolution`: True
   - `EnableDnsSupport`: True
2. **Service Discovery**: Gateway service must be registered with Cloud Map
3. **Security Groups**: Both services must share the same security group with ingress rules for:
   - Port 5672 (RabbitMQ)
   - Port 6379 (Redis)
   - Port 8000 (Chroma)
4. **Port Mapping**: Gateway containers must have port mappings (already configured)

**Fix**:
```bash
# Enable VPC DNS
python scripts/fix-vpc-dns.py

# Restart agent service to pick up DNS changes
aws ecs update-service --cluster euglena-cluster --service euglena-agent --force-new-deployment --region us-east-2
```

**Verification**:
```bash
# Diagnose connectivity
python scripts/diagnose-agent-connectivity.py

# Check VPC DNS
aws ec2 describe-vpcs --vpc-ids <vpc-id> --query 'Vpcs[0].[EnableDnsHostnames,EnableDnsSupport]'
```

## Queue Behavior

### Expected Queue Depth

**Queue depth of 0 is normal** when the agent is actively consuming messages. This indicates:
- Agent is running and connected to RabbitMQ
- Messages are being processed successfully
- Processing rate exceeds or matches submission rate

### Skip Message Processing

Skip messages (`skipskipskip`) are processed quickly:
- **Processing time**: ~10 seconds per message (includes connectivity test + delay)
- **Queue behavior**: With 1 agent instance, queue depth will be 0-1 during normal operation
- **Batch processing**: 30 skip messages would be processed in ~5 minutes (30 × 10s = 300s)

### Verifying Message Processing

To verify messages are being processed:
1. Check agent logs for "SKIP MODE" entries
2. Check gateway logs for task submissions
3. Monitor queue depth over time (should fluctuate between 0-1 with single agent)
4. Check Redis for task status updates

**Note**: If queue depth stays at 0 and no agent logs appear, the agent may not be consuming messages (check connectivity).

## Cluster Configuration

### Container Insights
- **Status**: Enabled (required for task-level metrics in CloudWatch)
- **Configuration**: Automatically enabled by `ensure_cluster()` in deployment scripts
- **Note**: Must be enabled to view task-level metrics in CloudWatch console

## Environment Variables

### Gateway Container
- `GATEWAY_DEBUG_QUEUE_NAME`: `gateway.debug` (configurable via .env)
- `GATEWAY_DEBUG_QUEUE_PHRASE`: `debugdebugdebug` (configurable via .env)
- `REDIS_URL`: `redis://localhost:6379/0`
- `CHROMA_URL`: `http://localhost:8000`
- `RABBITMQ_URL`: `amqp://guest:guest@localhost:5672/`

### Metrics Container
- `QUEUE_NAME`: `agent.mandates`
- `GATEWAY_DEBUG_QUEUE_NAME`: `gateway.debug` (from ConnectorConfig)
- `QUEUE_DEPTH_METRICS_INTERVAL`: `1` (1 second)
- `PUBLISH_QUEUE_DEPTH_METRICS`: `true`
- `RABBITMQ_URL`: `amqp://guest:guest@localhost:5672/`
- `CLOUDWATCH_NAMESPACE`: `Euglena/RabbitMQ`

## Notes

- **Metrics container is non-essential**: If it fails, the task will continue running
- **Gateway depends on Redis and RabbitMQ**: These must be healthy for gateway to start
- **Service discovery**: Gateway must be registered for agent to connect
- **ALB health checks**: Run every 30 seconds, 3 failures = unhealthy
- **Container health checks**: Run independently of ALB health checks
- **Autoscale mode**: Agent and gateway are in separate tasks; agent connects via service discovery DNS
- **VPC DNS**: MUST be enabled for service discovery to work in autoscale mode
- **Container Insights**: MUST be enabled for task-level metrics
- **Memory allocation**: 2048 MB provides ~409 MB per container, sufficient for RabbitMQ
- **Queue creation**: Metrics service automatically creates `gateway.debug` queue if it doesn't exist

## Historical Context

- **r46 (2026-02-01)**: Stable configuration - 512 CPU, 1024 memory, 5 containers
- **r49 (2026-02-03)**: Missing metrics container - caused failures
- **r54 (2026-02-04)**: Doubled resources - still failing
- **r56 (2026-02-04)**: Reverted to stable r46 config - stable
- **r68+ (2026-02-05)**: Increased to 1024 CPU, 2048 MB to prevent RabbitMQ OOM - current stable

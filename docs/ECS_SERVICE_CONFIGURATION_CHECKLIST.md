# ECS Service Configuration Checklist

This checklist documents required configuration changes for ECS services based on operational observations and troubleshooting.

## Gateway Service (euglena-gateway)

### Critical Issues

- [ ] **Desired Tasks: 1 → 2**
  - **Current:** 1 task (single point of failure)
  - **Required:** Minimum 2 tasks for high availability
  - **Impact:** Service unavailable if single task fails

- [ ] **Health Check Grace Period: 360s → 60-90s**
  - **Current:** 360 seconds (6 minutes)
  - **Required:** 60-90 seconds
  - **Impact:** Unhealthy tasks remain in service too long, causing request failures
  - **Reason:** Gateway connects to RabbitMQ and Redis in 10-30 seconds typically

- [ ] **Subnets: Mixed → Public Only**
  - **Current:** Mix of public and private subnets
  - **Required:** Public subnets only:
    - `subnet-059a0845ba7eb4a09` (euglena-prod-public-us-east-2a)
    - `subnet-0cd121febce7985b9` (euglena-prod-public-us-east-2b)
  - **Impact:** Gateway needs direct internet access for Supabase authentication
  - **Note:** Remove private subnets from configuration

### Recommended Changes

- [ ] **Service Auto Scaling: Disabled → Enabled**
  - **Min capacity:** 2
  - **Max capacity:** 5-10
  - **Target:** 500-1000 requests per target
  - **Scale-in cooldown:** 300 seconds
  - **Scale-out cooldown:** 60 seconds

- [ ] **Target Group Health Check Settings**
  - **Health check interval:** 30 seconds
  - **Healthy threshold:** 2 consecutive successes
  - **Unhealthy threshold:** 3 consecutive failures
  - **Timeout:** 5 seconds
  - **Path:** `/health`

---

## Agent Service (euglena-agent)

### Critical Issues

- [ ] **Desired Tasks: 0 → 1 (Minimum)**
  - **Current:** 0 (service scaled to zero)
  - **Required:** Minimum 1 (matches MIN_WORKERS=1 in autoscaling lambda)
  - **Impact:** No agents running, tasks cannot be processed
  - **Status:** Autoscaling lambda sets MIN_WORKERS=1, but service must be at least 1 for it to work
  - **Note:** When queue depth is 0, autoscaling correctly scales to MIN_WORKERS=1

- [ ] **Security Group: gateway-sg → agent-sg (NEW)**
  - **Current:** Using `sg-0c45255737f0f9199` (euglena-gateway-sg) - WRONG
  - **Required:** Create dedicated agent security group
  - **Required Rules:**
    - **Outbound:** All traffic (0.0.0.0/0) for:
      - RabbitMQ (internal VPC)
      - Redis (internal VPC)
      - ChromaDB (internal VPC)
      - LLM API (external internet)
      - Search API (external internet)
      - Web scraping (HTTP connector)
    - **Inbound:** None (agents don't receive connections)
  - **Impact:** Using gateway security group is incorrect and may cause connectivity issues

- [ ] **Service Auto Scaling: Not Configured → Configured**
  - **Min capacity:** 1 (matches MIN_WORKERS)
  - **Max capacity:** 11 (matches MAX_WORKERS)
  - **Note:** Lambda autoscaling function updates desired count directly, but ECS auto-scaling provides better integration
  - **Status:** Lambda autoscaling is working (observed: requested 3 workers, only 1 running with 2 pending when queue=0, MIN_WORKERS=1)

### Important Considerations

- [ ] **Public IP: Enabled (Verify NAT Gateway)**
  - **Current:** Enabled with private subnets
  - **Status:** Correct if NAT gateway is configured
  - **Required:** Agents need outbound internet for:
    - LLM API calls
    - Search API calls
    - Web scraping
  - **Action:** Verify NAT gateway is configured for private subnets, or keep Public IP enabled

- [ ] **Subnets: Private Only**
  - **Current:** Private subnets only
  - **Status:** CORRECT - Agents don't need direct inbound internet access
  - **Note:** Agents consume from RabbitMQ internally, no need for public subnets

### Configuration Status

- [x] **Health Check Grace Period: 60 seconds** - CORRECT
- [x] **Service Discovery: Disabled** - CORRECT (agents don't need service discovery)
- [x] **Load Balancing: Disabled** - CORRECT (agents don't serve HTTP traffic)

---

## Autoscaling Behavior Observations

### Current Behavior (Working as Designed)

**Observation:** Requested 3 agent workers, but only 1 is running (2 pending)

**Analysis:**
- Queue depth: 0
- MIN_WORKERS: 1 (configured in lambda autoscaling)
- Autoscaling logic: `if queue_depth == 0: return MIN_WORKERS`
- Result: Correctly scales to 1 worker (minimum)

**Status:** Autoscaling is working correctly

### Known Issue: Queue Length Reporting

**Problem:** Queue length reporting appears "iffy" - metrics service may have difficulty seeing items in queue

**Symptoms:**
- Metrics service reports queue depth as 0 even when tasks are submitted
- Gateway successfully publishes to RabbitMQ
- Tasks may be in queue but not visible to metrics/autoscaling

**Potential Causes:**
1. **Metrics service connection issues:**
   - RabbitMQ connection not ready
   - Queue depth check failing silently
   - Connection state not properly tracked

2. **Timing issues:**
   - Metrics checks queue before message is fully enqueued
   - CloudWatch metric publishing delay
   - Queue depth metric not published frequently enough

3. **Queue name mismatch:**
   - Metrics checking wrong queue name
   - Environment variable mismatch between services

**Investigation Steps:**
- [ ] Verify metrics service RabbitMQ connection status
- [ ] Check metrics service logs for queue depth check failures
- [ ] Verify queue name matches across all services (AGENT_INPUT_QUEUE)
- [ ] Check CloudWatch metrics for QueueDepth metric
- [ ] Verify metrics service is publishing to correct CloudWatch namespace
- [ ] Check timing between task submission and queue depth check

**Files to Check:**
- `services/metrics/app/main.py` - Queue depth checking logic
- `services/shared/connector_rabbitmq.py` - Queue depth implementation
- `services/shared/queue_metrics.py` - CloudWatch metric configuration
- Gateway logs - Verify successful publish
- Metrics service logs - Queue depth check results

---

## Implementation Priority

### Immediate (Service Unavailable)
1. Agent Service: Set Desired Tasks to 1
2. Agent Service: Create and assign correct security group
3. Gateway Service: Increase Desired Tasks to 2

### High Priority (Service Degradation)
4. Gateway Service: Reduce Health Check Grace Period
5. Gateway Service: Fix subnet configuration (public only)
6. Agent Service: Configure service auto-scaling

### Medium Priority (Optimization)
7. Gateway Service: Enable service auto-scaling
8. Investigate queue length reporting issues

---

## Verification Commands

### Check Gateway Service
```bash
aws ecs describe-services \
  --cluster euglena-cluster \
  --services euglena-gateway \
  --query 'services[0].{DesiredCount:desiredCount,HealthCheckGracePeriod:healthCheckGracePeriodSeconds,Subnets:networkConfiguration.awsvpcConfiguration.subnets}'
```

### Check Agent Service
```bash
aws ecs describe-services \
  --cluster euglena-cluster \
  --services euglena-agent \
  --query 'services[0].{DesiredCount:desiredCount,SecurityGroups:networkConfiguration.awsvpcConfiguration.securityGroups,Subnets:networkConfiguration.awsvpcConfiguration.subnets}'
```

### Check Queue Depth Metrics
```bash
aws cloudwatch get-metric-statistics \
  --namespace Euglena/QueueMetrics \
  --metric-name QueueDepth \
  --dimensions Name=QueueName,Value=agent.mandates \
  --start-time $(date -u -d '1 hour ago' -Iseconds) \
  --end-time $(date -u -Iseconds) \
  --period 60 \
  --statistics Average
```

---

## Last Updated
- Date: 2025-01-13
- Status: Autoscaling working correctly, queue length reporting needs investigation
- Agent Service: 1 worker running (correct for queue depth 0, MIN_WORKERS 1)

---
name: Autoscale Redux Migration Plan
overview: Incremental migration plan to recreate autoscaling functionality without the instability issues. Starts by recreating the script system to match stable manual deployment, then gradually ports autoscaling features in testable stages.
todos:
  - id: stage0-script-system
    content: "Stage 0: Create script system with OOP modules (NetworkDiscovery, EcsInfrastructure), deploy.py that works from scratch (not just updates), VPC auto-discovery, and requirements.txt. All subscripts can be run directly or imported."
    status: completed
  - id: stage1-infrastructure
    content: "Stage 1: Port infrastructure improvements (queue_metrics.py definitions) without changing deployment. Includes stable health checks, Chroma health check in agent, and embedding model caching on EFS."
    status: in_progress
    dependencies:
      - stage0-script-system
  - id: stage2-service-separation
    content: "Stage 2: Split into gateway and agent services (fixed agent count=1, no autoscaling yet)"
    status: pending
    dependencies:
      - stage1-infrastructure
  - id: stage3-metrics-service
    content: "Stage 3: Add metrics service to gateway task, publish queue depth to CloudWatch"
    status: pending
    dependencies:
      - stage2-service-separation
  - id: stage4-lambda-autoscaling
    content: "Stage 4: Add Lambda autoscaling function (manual deployment for testing first)"
    status: pending
    dependencies:
      - stage3-metrics-service
  - id: stage5-ecs-manager
    content: "Stage 5: Add ECS manager and task protection to prevent agents from being killed mid-task"
    status: pending
    dependencies:
      - stage4-lambda-autoscaling
  - id: stage6-service-discovery
    content: "Stage 6: Configure AWS Cloud Map service discovery for agent-to-gateway communication"
    status: pending
    dependencies:
      - stage5-ecs-manager
  - id: stage7-full-autoscaling
    content: "Stage 7: Enable full autoscaling with EventBridge triggers and conservative limits"
    status: pending
    dependencies:
      - stage6-service-discovery
  - id: stage8-superdeploy
    content: "Stage 8: Create superdeploy.py script with 30-minute stability testing and comprehensive monitoring"
    status: pending
    dependencies:
      - stage7-full-autoscaling
  - id: stage9-supabase-tracking
    content: "Stage 9: Reintroduce improved Supabase task tracking system (StorageManager, Supabase as source of truth, multiple concurrent tasks per user)"
    status: pending
    dependencies:
      - stage8-superdeploy
---

# Autoscale Redux Migration Plan

## Overview

This plan migrates autoscaling functionality from the unstable `autoscale/` version to a new `autoscale-redux` branch, incorporating features incrementally to identify and avoid the instability issues. The migration follows a staged approach where each stage is tested before proceeding.

## Architecture Comparison

### Current Stable (Main Branch)

- **Single ECS Service**: One task definition with all containers (gateway, agent, redis, rabbitmq, chroma)
- **Manual Deployment**: `scripts/build-task-definition.py` generates single task definition
- **No Autoscaling**: Fixed single service instance
- **Stable**: Production-ready, no known issues

### Autoscale Version (Unstable)

- **Separate Services**: Gateway service (gateway, redis, rabbitmq, chroma, metrics) + Agent service (scalable)
- **Automated Deployment**: `autoscale/scripts/deploy.py` handles full deployment pipeline
- **Autoscaling**: Lambda function scales agent tasks based on RabbitMQ queue depth
- **Service Discovery**: Agent tasks discover gateway via AWS Cloud Map
- **Metrics**: Queue depth published to CloudWatch
- **Improved Task Tracking**: Supabase as source of truth, StorageManager for sync, multiple concurrent tasks per user
- **Frontend/Gateway Changes**: Gateway writes to both Redis and Supabase, syncs Redis→Supabase, frontend reads only from Supabase
- **Issues**: High instability, silent failures (but task tracking system is good)

## Migration Strategy

### Stage 0: Script System Recreation (Foundation)

**Goal**: Create deployment scripts that work from scratch (not just updates) and produce identical results to stable manual deployment

**Tasks**:

1. Create `autoscale-redux` branch from main
2. Create `scripts/requirements.txt` with dependencies:

   - `boto3` (AWS SDK)
   - `python-dotenv` (environment variable loading)
   - Other dependencies as needed

3. Create OOP-based network discovery module `scripts/network_discovery.py`:

   - `NetworkDiscovery` class to find VPCs, subnets, security groups
   - Methods: `find_vpc()`, `find_subnets()`, `find_security_groups()`
   - Can be run directly: `python scripts/network_discovery.py` (prints discovered network info)
   - Can be imported: `from scripts.network_discovery import NetworkDiscovery`

4. Create OOP-based ECS infrastructure module `scripts/ecs_infrastructure.py`:

   - `EcsInfrastructure` class to manage ECS resources
   - Methods: `ensure_cluster()`, `ensure_service()`, `create_or_update_service()`
   - Handles both creation (first deploy) and updates (subsequent deploys)
   - Can be run directly or imported

5. Port and enhance `autoscale/scripts/deploy.py` to `scripts/deploy.py`:

   - Build single task definition (like stable version)
   - Use existing `scripts/build-task-definition.py` logic
   - Support ECR push, task definition registration
   - **NEW**: Use `EcsInfrastructure` to create services if they don't exist
   - **NEW**: Use `NetworkDiscovery` to find VPCs/subnets/security groups automatically
   - Skip autoscaling features (Lambda, EventBridge, service discovery) for now
   - Handle first-time deployment (create cluster, create services, create security groups if needed)

6. Port `autoscale/scripts/check.py` to `scripts/check.py` for health monitoring
7. Port and enhance `autoscale/scripts/network_utils.py` to `scripts/network_utils.py`:

   - Use `NetworkDiscovery` class internally
   - Can be run directly for network validation
   - Can be imported for use in other scripts

8. Port `autoscale/scripts/fix-iam-permissions.py` to `scripts/fix-iam-permissions.py`
9. Test: Deploy from scratch (no existing services), verify it creates everything
10. Test: Deploy again (existing services), verify it updates correctly

**Files to Create/Modify**:

- `scripts/requirements.txt` (new, contains: boto3, python-dotenv, and other dependencies)
- `scripts/network_discovery.py` (new, OOP-based VPC/subnet discovery)
- `scripts/ecs_infrastructure.py` (new, OOP-based ECS resource management)
- `scripts/deploy.py` (new, uses OOP modules, handles first-time deployment)
- `scripts/check.py` (new)
- `scripts/network_utils.py` (enhanced, uses NetworkDiscovery)
- `scripts/fix-iam-permissions.py` (new)

**Scripts Requirements.txt Content**:

```
boto3>=1.28.0
python-dotenv>=1.0.0
```

**Architecture**:

```
scripts/
├── requirements.txt              # boto3, python-dotenv, etc.
├── network_discovery.py          # NetworkDiscovery class (VPC/subnet finding)
├── ecs_infrastructure.py         # EcsInfrastructure class (cluster/service management)
├── deploy.py                     # Main deployment (uses above classes)
├── check.py                      # Health checks
├── network_utils.py              # Network validation (uses NetworkDiscovery)
└── fix-iam-permissions.py        # IAM setup
```

**Key Features**:

- **First-time deployment**: Creates ECS cluster, services, security groups, task definitions
- **Update deployment**: Updates existing services with new task definitions
- **VPC discovery**: Automatically finds VPCs/subnets (can be configured or auto-discovered)
- **Modular design**: Each subscript can be run directly or imported
- **OOP where appropriate**: NetworkDiscovery, EcsInfrastructure classes for reusable logic

**Success Criteria** - ALL MET:

- `pip install -r scripts/requirements.txt` installs all dependencies - COMPLETED
- `python scripts/network_discovery.py` discovers and prints VPC info - COMPLETED
- `python scripts/deploy.py` works on fresh AWS account (no existing services) - COMPLETED
- `python scripts/deploy.py` works on existing deployment (updates services) - COMPLETED
- Deployment produces identical results to manual process - COMPLETED
- All services run stably for 30+ minutes - COMPLETED
- Health checks pass - COMPLETED
- ALB integration configured - COMPLETED
- Service name matches specification (euglena-service) - COMPLETED

---

### Stage 0.5: Agent Skip Mode for Connectivity Testing

**Goal**: Add special skip mode to agent for easier debugging - allows testing connectivity without running full agent logic

**Tasks**:

1. **Add Skip Phrase Detection**:

   - Add environment variable `AGENT_SKIP_PHRASE` (default: `skipskipskip`)
   - Check if mandate contains skip phrase in `_handle_task()` method
   - If skip phrase found, skip `agent.run()` entirely

2. **Add Connectivity Testing**:

   - Create `_test_connectivity()` method in `InterfaceAgent`
   - Test RabbitMQ connection (check `rabbitmq.is_ready()`)
   - Test Redis connection (ping Redis)
   - Test Chroma connection (check `connector_chroma.chroma_api_ready`)
   - Test external API access (make HTTP request to external URL like google.com)
   - Return connectivity status dictionary

3. **Return Default Response**:

   - When skip phrase detected, return default response with connectivity test results
   - Response should include:
     - Success status (all services connected)
     - Connectivity status for each service (RabbitMQ, Redis, Chroma, External API)
     - Clear message indicating skip mode was used

4. **Update Environment Variables**:

   - Document `AGENT_SKIP_PHRASE` in environment variable documentation
   - Default value: `skipskipskip`

5. **Testing**:

   - Submit task with skip phrase in mandate
   - Verify agent skips `agent.run()`
   - Verify connectivity test runs
   - Verify response contains connectivity status
   - Test with each service disconnected to verify failure detection

**Files to Modify**:

- `services/agent/app/interface_agent.py` (add skip detection and connectivity testing)
- Environment variable documentation

**Usage**:

To test connectivity, submit a task with the skip phrase in the mandate:

```
mandate: "skipskipskip test connectivity"
```

The agent will:

1. Accept the task
2. Test connectivity to RabbitMQ, Redis, Chroma, and external APIs
3. Return a response with connectivity status
4. Skip the full agent execution

**Success Criteria**:

- Skip phrase detection works correctly
- Connectivity test checks all required services
- Default response returned when skip phrase detected
- Connectivity status accurately reflects service availability
- Agent can connect to RabbitMQ, Chroma, Redis, and external APIs when all are available
- Agent correctly detects when services are unavailable

---

### Stage 1: Infrastructure Improvements and Stable Health Checks

**Goal**: Port stable infrastructure utilities and improve health checks without changing deployment architecture

**Tasks**:

1. Port `autoscale/services/shared/queue_metrics.py` (just the definitions, no publishing yet)
2. Port improved error handling patterns from autoscale (if they're better)
3. **Add stable health checks** (COMPLETED):

   - Review and improve health check configurations in task definitions - COMPLETED
   - Ensure health checks are lenient enough to prevent false failures - COMPLETED
   - Add proper start periods, intervals, retries, and timeouts - COMPLETED
   - Test health checks don't cause unnecessary task restarts - COMPLETED
   - **Added Chroma health check to agent**: Agent health endpoint now includes Chroma connectivity status - COMPLETED
   - Chroma health status now visible in ECS console (no longer shows "Unknown") - COMPLETED

4. **Fix 403 errors**: Add User-Agent header to HTTP connector to avoid bot detection - COMPLETED
5. **Bake embedding model** (COMPLETED):

   - Pre-download Chroma embedding model in Docker image to avoid runtime downloads - COMPLETED
   - Configured agent container with `SENTENCE_TRANSFORMERS_HOME` and `TRANSFORMERS_CACHE` environment variables - COMPLETED
   - Configured Chroma container to cache models on EFS (`/chroma-data/.cache`) for persistence - COMPLETED
   - Models now persist across container restarts (cached on EFS) - COMPLETED
   - Prevents unnecessary model re-downloads on each container restart - COMPLETED

6. Test: Verify no regressions, services remain stable

**Files to Port/Modify**:

- `services/shared/queue_metrics.py` (metric definitions only)
- `services/agent/app/connector_http.py` (add User-Agent header) - COMPLETED
- `services/agent/.dockerfile` (pre-download embedding model, set cache env vars) - COMPLETED
- `services/agent/app/main.py` (add Chroma health check) - COMPLETED
- `scripts/build-task-definition.py` (improve health check configurations, add Chroma cache env vars) - COMPLETED

**Success Criteria**:

- No functionality changes - COMPLETED
- Health checks are stable and don't cause false failures - COMPLETED
- 403 errors reduced with proper User-Agent headers - COMPLETED
- Embedding model pre-downloaded in image (no runtime downloads) - COMPLETED
- Chroma health check added to agent health endpoint - COMPLETED
- Models cached on EFS for persistence across restarts - COMPLETED
- Services remain stable for 30+ minutes

---

### Stage 2: Service Separation with Volume Mounts

**Goal**: Split into gateway and agent services with persistent storage, but keep agent at fixed count (1)

**Tasks**:

1. **Set up EFS volumes** (COMPLETED):

   - EFS file system configured: `fs-0ec151e2adb754fc8` (shared across all services)
   - EFS mount targets automatically created in specified subnets
   - Security group rules automatically configured to allow ECS task access
   - Root directories configured: `/chroma-data`, `/redis-data`, `/rabbitmq-data`
   - Transit encryption enabled for all volumes
   - IAM authorization disabled when using `rootDirectory` (requires access points otherwise)
   - `scripts/deploy.py` automatically creates mount targets and updates security groups
   - `scripts/build-task-definition.py` configures volume mounts in all task definitions
   - `scripts/efs_manager.py` handles mount target creation and security group management

2. Modify `scripts/build-task-definition.py` to build two task definitions (COMPLETED):

   - `euglena-gateway`: gateway, redis, rabbitmq, chroma containers - COMPLETED
     - Chroma: Mount EFS volume to `/chroma-data` - CONFIGURED
     - Redis: EFS volume mount at `/redis-data` - CONFIGURED
     - RabbitMQ: EFS volume mount at `/rabbitmq-data` - CONFIGURED
   - `euglena-agent`: agent container only - COMPLETED
   
   **Recent Fixes (January 2026)**:
   - Fixed EFS mount access denied errors by automatically configuring security group rules
   - Fixed IAM authorization incompatibility with `rootDirectory` (disabled when using root directories)
   - Enhanced EFS manager to detect and update mount target security groups automatically
   - Improved deployment script to ensure ECS task security groups can access EFS on every deployment

3. Update `scripts/deploy.py` to:

   - Build both task definitions
   - Create/update two ECS services
   - EFS file system and mount targets automatically created/verified (COMPLETED)
   - Security group rules automatically updated to allow ECS access (COMPLETED)
   - Keep agent service at desiredCount=1 (fixed)

4. Update agent to connect to gateway via localhost (same task) → service discovery DNS
5. Test: Deploy both services, verify agent can reach gateway services, verify data persists

**Files to Modify**:

- `scripts/build-task-definition.py` (split into `build_gateway_task_definition()` and `build_agent_task_definition()`, add volume mounts)
- `scripts/deploy.py` (handle two services, create EFS resources if needed)
- `scripts/ecs_infrastructure.py` (add EFS file system creation/verification)
- Agent environment variables (update connection URLs)

**Volume Configuration** (COMPLETED):

- **Chroma**: EFS mount at `/chroma-data` (required for persistence) - CONFIGURED
- **Redis**: EFS mount at `/redis-data` (persistence enabled) - CONFIGURED
- **RabbitMQ**: EFS mount at `/rabbitmq-data` (persistence enabled) - CONFIGURED
- All volumes share the same EFS file system (`fs-0ec151e2adb754fc8`) with separate root directories
- Transit encryption enabled for all volumes
- Security group access automatically configured on deployment
- IAM authorization disabled when using `rootDirectory` (requires access points otherwise)

**Success Criteria**:

- Both services deploy successfully
- Agent can connect to gateway's redis/rabbitmq/chroma
- Chroma data persists across task restarts (EFS working)
- Services remain stable for 30+ minutes
- No autoscaling active (fixed at 1 agent)

---

### Stage 3: Metrics Service

**Goal**: Add metrics service to gateway task, publish queue depth to CloudWatch

**Tasks**:

1. Port `autoscale/services/metrics/` to `services/metrics/`
2. Add metrics container to gateway task definition
3. Configure metrics service to publish to CloudWatch
4. Test: Verify metrics appear in CloudWatch, no service instability

**Files to Port/Create**:

- `services/metrics/app/main.py`
- `services/metrics/requirements.txt`
- Update gateway task definition to include metrics container

**Success Criteria**:

- Metrics service runs in gateway task
- Queue depth metrics appear in CloudWatch
- No increase in errors or instability
- Services stable for 30+ minutes

---

### Stage 4: Lambda Autoscaling Function

**Goal**: Add Lambda function to read CloudWatch metrics and scale agent service, **with fixes for RabbitMQ tracking issues**

**Known Issue**: The autoscale version's Lambda can scale workers successfully, but doesn't track RabbitMQ queue depth correctly. This needs to be fixed.

**Tasks**:

1. Port `autoscale/services/lambda_autoscaling/` to `services/lambda_autoscaling/`
2. **Debug RabbitMQ Tracking Issue**:

   - Verify metrics service is publishing to CloudWatch correctly
   - Check namespace, metric name, and dimensions match between metrics service and Lambda
   - Verify CloudWatch query parameters (time window, period, statistics)
   - Test direct CloudWatch queries to see if metrics exist
   - Add detailed logging to Lambda for metric retrieval
   - Compare actual queue depth (from RabbitMQ management API) with CloudWatch metrics

3. **Fix RabbitMQ Tracking**:

   - Ensure namespace matches: `Euglena/RabbitMQ` (or configured value)
   - Ensure metric name matches: `QueueDepth`
   - Ensure dimensions match: `QueueName=agent.mandates` (or configured queue)
   - Adjust time window if needed (currently 2 minutes, metrics published every 5 seconds)
   - Consider using `Sum` or `Maximum` statistics instead of `Average` if needed
   - Add retry logic for CloudWatch queries
   - Add fallback to query RabbitMQ directly if CloudWatch fails (optional)

4. Update `scripts/deploy.py` to package Lambda (but don't deploy it yet)
5. Manually deploy Lambda function for testing
6. Create EventBridge rule manually for testing
7. Test: Verify Lambda can read metrics correctly and update service
8. Test: Submit tasks, verify queue depth is tracked correctly in CloudWatch
9. Test: Verify Lambda scales based on actual queue depth (not just fallback)
10. Monitor for 30 minutes for instability

**Potential Issues to Investigate**:

- **Namespace mismatch**: Metrics service and Lambda using different namespaces
- **Dimension mismatch**: Queue name not matching between publisher and consumer
- **Timing issues**: CloudWatch metrics may have delay, Lambda querying too early
- **Statistics selection**: Using `Average` might not reflect current queue depth accurately
- **Time window**: 2-minute window might miss recent metrics
- **Metrics not publishing**: Metrics service might not be publishing correctly

**Files to Port/Modify**:

- `services/lambda_autoscaling/lambda_function.py` (port and fix RabbitMQ tracking)
- `services/lambda_autoscaling/requirements.txt`
- Update `scripts/deploy.py` `package_lambda()` function
- Add debugging utilities if needed

**Success Criteria**:

- Lambda function packages correctly
- Lambda can read CloudWatch metrics **correctly** (queue depth matches actual RabbitMQ queue)
- Lambda scales based on actual queue depth (not just fallback to MIN_WORKERS)
- Lambda can update ECS service desiredCount
- Queue depth tracking is accurate and reliable
- No immediate instability (30-minute test)

---

### Stage 5: ECS Manager & Task Protection

**Goal**: Add ECS task protection to prevent agents from being killed during work

**Tasks**:

1. Port `autoscale/services/shared/ecs_manager.py` to `services/shared/ecs_manager.py`
2. Integrate ECS manager into agent (`services/agent/app/interface_agent.py`)
3. Enable task protection when agent starts processing task
4. Disable task protection when task completes
5. Test: Verify task protection works, agents aren't killed mid-task

**Files to Port/Modify**:

- `services/shared/ecs_manager.py`
- `services/agent/app/interface_agent.py` (integrate ECS manager)
- `services/agent/app/main.py` (initialize ECS manager)

**Success Criteria**:

- Task protection enables/disables correctly
- Agents complete tasks without interruption
- No increase in errors
- Services stable for 30+ minutes

---

### Stage 6: Service Discovery

**Goal**: Configure AWS Cloud Map so agent tasks can discover gateway via DNS

**Tasks**:

1. Port `setup_service_discovery()` from `autoscale/scripts/deploy.py`
2. Update agent task definition to use service discovery DNS names
3. Update agent environment variables to use `euglena-gateway.euglena.local` instead of localhost
4. Test: Deploy with service discovery, verify agent can resolve and connect

**Files to Modify**:

- `scripts/deploy.py` (add `setup_service_discovery()` call)
- `scripts/build-task-definition.py` (update agent env vars for service discovery)
- Agent connection URLs

**Success Criteria**:

- Service discovery configured
- Agent tasks resolve gateway DNS correctly
- All connections work via service discovery
- Services stable for 30+ minutes

---

### Stage 7: Full Autoscaling Integration

**Goal**: Enable full autoscaling with Lambda triggered by EventBridge

**Tasks**:

1. Integrate `setup_autoscaling_rule()` into `scripts/deploy.py`
2. Configure EventBridge to trigger Lambda every 1-2 minutes
3. Set MIN_WORKERS=1, MAX_WORKERS=5 (conservative limits)
4. Test: Submit tasks, verify agents scale up/down
5. Extended test: 30-minute stability monitoring

**Files to Modify**:

- `scripts/deploy.py` (enable autoscaling rule setup)
- Lambda configuration (schedule, limits)

**Success Criteria**:

- Autoscaling triggers correctly
- Agents scale up when queue has tasks
- Agents scale down when queue is empty
- No instability over 30 minutes
- No silent failures

---

### Stage 8: Superdeploy Script

**Goal**: Create comprehensive deployment script with extended stability testing

**Tasks**:

1. Create `scripts/superdeploy.py` that:

   - Runs full deployment (`deploy.py`)
   - Waits for services to stabilize
   - Runs health checks every 2 minutes for 30 minutes
   - Monitors:
     - ECS service stability (no unexpected restarts)
     - Container health (all containers healthy)
     - Error rates in CloudWatch logs
     - Task completion rates
     - Queue depth trends
   - Generates stability report

2. Add alerting for:

   - Service restarts
   - Health check failures
   - High error rates
   - Task failures

**Files to Create**:

- `scripts/superdeploy.py` (new comprehensive deployment + testing script)

**Success Criteria**:

- Superdeploy script runs successfully
- Detects instability issues automatically
- Generates clear stability reports
- Can be used for all future deployments

---

### Stage 9: Improved Supabase Task Tracking System

**Goal**: Reintroduce the improved task tracking system where Supabase is the source of truth, enabling users to have multiple concurrent tasks

**Background**:

The autoscale version modified the gateway and frontend to use Supabase as the primary storage for task status and responses. This allows:

- Users to have multiple tasks running concurrently
- Better persistence (tasks survive Redis restarts)
- Clear separation: workers write to Redis, gateway syncs to Supabase
- Frontend only reads from Supabase (source of truth)

**Current State (Main Branch)**:

- Tasks stored primarily in Redis
- Limited concurrent task support
- Less persistence

**Target State (Autoscale Version)**:

- Supabase is source of truth for frontend
- Gateway writes to both Redis (for workers) and Supabase (for persistence)
- Gateway syncs Redis → Supabase when reading tasks
- Frontend only reads from Supabase
- Users can have many concurrent tasks

**Tasks**:

1. **Port Supabase Schema**:

   - Port `autoscale/services/supabase/schema.sql` to `services/supabase/schema.sql`
   - Port `autoscale/services/supabase/profiles_rls_fix.sql` to `services/supabase/profiles_rls_fix.sql`
   - Create migration script to apply schema to existing Supabase instance
   - Document schema differences from main branch

2. **Port StorageManager**:

   - Port `autoscale/services/gateway/app/storage_manager.py` to `services/gateway/app/storage_manager.py`
   - This class handles:
     - Creating tasks in both Redis and Supabase
     - Reading from Supabase (source of truth)
     - Syncing Redis → Supabase when Redis has newer updates
     - Automatic cleanup of completed tasks from Redis

3. **Update Gateway Service**:

   - Port modified `autoscale/services/gateway/app/gateway_service.py` logic
   - Integrate StorageManager for all task operations
   - Update task creation to write to both Redis and Supabase
   - Update task retrieval to read from Supabase and sync from Redis
   - Update task listing to read from Supabase only

4. **Update Gateway API**:

   - Port modified `autoscale/services/gateway/app/api.py` if needed
   - Ensure all endpoints use StorageManager
   - Verify authentication and RLS policies work correctly

5. **Update Frontend** (if needed):

   - Port any frontend changes from autoscale version
   - Ensure frontend can handle multiple concurrent tasks
   - Verify task listing and status polling work with Supabase

6. **Testing**:

   - Test task creation (should write to both Redis and Supabase)
   - Test task status updates (agent → Redis → gateway syncs to Supabase)
   - Test task retrieval (should read from Supabase, sync if needed)
   - Test multiple concurrent tasks per user
   - Test task persistence (restart Redis, tasks should still be in Supabase)
   - Verify RLS policies work correctly

**Files to Port/Create**:

- `services/supabase/schema.sql` (new, from autoscale)
- `services/supabase/profiles_rls_fix.sql` (new, from autoscale)
- `services/supabase/migration_guide.md` (new, documents migration steps)
- `services/gateway/app/storage_manager.py` (new, from autoscale)
- `services/gateway/app/gateway_service.py` (modify to use StorageManager)
- `services/gateway/app/api.py` (modify if needed)
- Frontend files (modify if needed)

**Key Changes**:

1. **Task Creation Flow**:
   ```
   User → Gateway → [Redis + Supabase] → RabbitMQ → Agent
   ```

2. **Task Status Update Flow**:
   ```
   Agent → Redis → Gateway detects → Gateway syncs Redis → Supabase → Frontend reads Supabase
   ```

3. **Task Retrieval Flow**:
   ```
   Frontend → Gateway → Read Supabase → Check Redis for newer updates → Sync if needed → Return Supabase data
   ```


**Migration Steps**:

1. Apply new Supabase schema (run SQL files)
2. Deploy updated gateway with StorageManager
3. Verify existing tasks are accessible
4. Test new task creation
5. Monitor sync behavior

**Success Criteria**:

- Supabase schema applied successfully
- Gateway uses StorageManager for all task operations
- Tasks are created in both Redis and Supabase
- Gateway syncs Redis → Supabase correctly
- Frontend can read multiple concurrent tasks from Supabase
- RLS policies enforce user isolation
- Tasks persist after Redis restarts
- No data loss during migration
- Services stable for 30+ minutes

---

## Testing Strategy

### Per-Stage Testing

- **Immediate**: Deploy and verify basic functionality
- **5-minute**: Check for obvious errors
- **30-minute**: Extended stability test (critical for catching silent failures)

### Stability Metrics

- **Service Restarts**: Count unexpected task stops
- **Health Check Failures**: Track health check success rate
- **Error Rates**: Monitor CloudWatch logs for errors
- **Task Completion**: Verify tasks complete successfully
- **Queue Depth**: Ensure queue is processed correctly

### Rollback Plan

- Each stage is a git commit
- Can rollback to previous stage if instability detected
- Keep stable version as reference

## Key Differences from Autoscale Version

### What We're Keeping (Good Features)

- Comprehensive deployment scripts
- Network validation utilities
- Health check scripts
- Metrics service architecture
- Lambda autoscaling logic
- Service discovery setup
- ECS task protection

### What We're Changing (Stability Fixes)

- Incremental migration (not all at once)
- Extended testing at each stage
- Conservative autoscaling limits initially
- Better error monitoring
- Rollback capability at each stage
- **Fix RabbitMQ queue tracking**: Lambda doesn't correctly track queue depth in autoscale version - will be fixed in Stage 4

### Potential Instability Sources to Watch

1. **Service Discovery DNS Resolution**: May cause connection issues
2. **Lambda Scaling Frequency**: Too frequent scaling may cause instability
3. **Task Protection**: May prevent proper scale-down
4. **Metrics Publishing**: May add load to gateway task
5. **Network Configuration**: Security group rules may be incorrect
6. **RabbitMQ Queue Tracking**: Lambda may not correctly read queue depth from CloudWatch (known issue in autoscale version - needs fixing in Stage 4)

## File Structure After Migration

```
euglena/
├── scripts/
│   ├── requirements.txt        # boto3, python-dotenv, etc.
│   ├── deploy.py              # Main deployment (stages 0-7)
│   ├── superdeploy.py         # Extended testing (stage 8)
│   ├── check.py               # Health checks
│   ├── build-task-definition.py  # Task definition builder
│   ├── network_discovery.py   # NetworkDiscovery class (VPC/subnet finding)
│   ├── ecs_infrastructure.py # EcsInfrastructure class (cluster/service management)
│   ├── network_utils.py       # Network validation (uses NetworkDiscovery)
│   └── fix-iam-permissions.py # IAM setup
├── services/
│   ├── gateway/               # Gateway service (with StorageManager in stage 9)
│   │   └── app/
│   │       └── storage_manager.py  # Supabase/Redis sync (stage 9)
│   ├── agent/                 # Agent service (with ECS manager)
│   ├── metrics/               # Metrics service (stage 3)
│   ├── lambda_autoscaling/    # Lambda function (stage 4)
│   ├── supabase/              # Supabase schema (stage 9)
│   │   ├── schema.sql         # Tasks table schema
│   │   └── profiles_rls_fix.sql  # RLS policies
│   └── shared/
│       ├── queue_metrics.py   # Metric definitions
│       └── ecs_manager.py     # Task protection (stage 5)
```

## Implementation Details for Stage 0

### NetworkDiscovery Class (`scripts/network_discovery.py`)

**Purpose**: Discover and manage AWS network resources (VPCs, subnets, security groups)

**Class Structure**:

```python
class NetworkDiscovery:
    def __init__(self, region: str, vpc_id: Optional[str] = None)
    def find_vpc(self, vpc_name: Optional[str] = None) -> Dict
    def find_subnets(self, vpc_id: str, subnet_names: Optional[List[str]] = None) -> List[Dict]
    def find_security_groups(self, vpc_id: str, sg_names: Optional[List[str]] = None) -> List[Dict]
    def get_default_vpc(self) -> Optional[Dict]
    def discover_all(self) -> Dict  # Returns complete network config
```

**Usage**:

- **Direct execution**: `python scripts/network_discovery.py` prints discovered network info
- **Import**: `from scripts.network_discovery import NetworkDiscovery`

**Features**:

- Auto-discovers VPCs if not specified (finds default VPC or by name)
- Finds subnets by name or returns all subnets in VPC
- Finds security groups by name or returns all in VPC
- Handles missing resources gracefully

### EcsInfrastructure Class (`scripts/ecs_infrastructure.py`)

**Purpose**: Manage ECS clusters and services (create or update)

**Class Structure**:

```python
class EcsInfrastructure:
    def __init__(self, region: str, cluster_name: str)
    def ensure_cluster(self) -> bool  # Creates if doesn't exist
    def ensure_service(self, service_name: str, task_family: str, 
                      network_config: Dict, desired_count: int = 1) -> bool
    def create_or_update_service(self, service_name: str, task_family: str,
                                network_config: Dict, desired_count: int) -> bool
    def service_exists(self, service_name: str) -> bool
```

**Usage**:

- **Direct execution**: `python scripts/ecs_infrastructure.py` manages ECS resources
- **Import**: `from scripts.ecs_infrastructure import EcsInfrastructure`

**Features**:

- Creates ECS cluster if it doesn't exist
- Creates ECS service if it doesn't exist (first-time deployment)
- Updates ECS service if it exists (subsequent deployments)
- Handles network configuration (VPC, subnets, security groups)
- Supports both Fargate and EC2 launch types

### Deploy Script (`scripts/deploy.py`)

**Enhanced Flow**:

1. Load AWS configuration from `aws.env`
2. Initialize `NetworkDiscovery` to find VPCs/subnets/security groups
3. Initialize `EcsInfrastructure` to manage ECS resources
4. Build and push Docker images to ECR
5. Build and register task definitions
6. Use `EcsInfrastructure.ensure_service()` to create or update services
7. Wait for services to stabilize

**Key Improvements**:

- Works on fresh AWS account (creates everything)
- Works on existing deployment (updates services)
- Auto-discovers network resources
- Modular design using OOP classes

## Success Criteria (Final)

1. Deployment scripts match stable manual deployment functionality
2. Services separated (gateway + agent) but stable
3. Metrics service publishing queue depth
4. Lambda autoscaling working correctly
5. Service discovery enabling agent-to-gateway communication
6. ECS task protection preventing mid-task kills
7. 30-minute stability tests passing
8. No silent failures or unexpected restarts
9. Superdeploy script providing comprehensive testing
10. Supabase task tracking system working (StorageManager, multiple concurrent tasks, persistence)

## Stage 0 Completion Status

Stage 0 has been completed successfully. All deployment scripts are operational and tested:

- Script system with OOP modules (NetworkDiscovery, EcsInfrastructure) - COMPLETED
- deploy.py works from scratch (creates everything) - COMPLETED
- deploy.py works for updates (updates existing services) - COMPLETED
- VPC auto-discovery - COMPLETED
- ALB integration - COMPLETED
- IAM permissions setup - COMPLETED
- Agent skip mode for connectivity testing (Stage 0.5) - COMPLETED

## Next Steps

1. Begin Stage 1: Port infrastructure improvements (queue_metrics.py definitions)
2. Test each stage before proceeding
3. Document any issues found during migration
4. Adjust plan based on findings
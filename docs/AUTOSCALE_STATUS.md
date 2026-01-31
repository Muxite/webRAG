# Autoscaling Migration Status

This document tracks the progress of migrating the Euglena project to full autoscaling capability. The migration follows an incremental staged approach to ensure stability at each step.

## Current State Summary

**Status**: Foundation complete, service separation in progress

The project has successfully completed the foundational stages (0, 0.5, and 1) and is currently working on service separation (Stage 2). The core deployment infrastructure is stable and operational.

## Completed Stages

### Stage 0: Script System Recreation (COMPLETED)

**Goal**: Create deployment scripts that work from scratch and produce identical results to stable manual deployment.

**Completed Components**:

- OOP-based network discovery module (`scripts/network_discovery.py`)
  - NetworkDiscovery class for VPC, subnet, and security group discovery
  - Can be run directly or imported as a module
  - Automatic VPC discovery with fallback to default VPC

- OOP-based ECS infrastructure module (`scripts/ecs_infrastructure.py`)
  - EcsInfrastructure class for cluster and service management
  - Handles both creation (first deploy) and updates (subsequent deploys)
  - Supports Fargate launch type with network configuration

- Main deployment script (`scripts/deploy.py`)
  - Works from scratch (creates everything on fresh AWS account)
  - Works for updates (updates existing services)
  - Automatic ECR image building and pushing
  - Image size calculation and reporting
  - Network validation and security group rule fixing
  - EFS file system management integration
  - ALB integration support

- Supporting scripts
  - `scripts/check.py` - Health monitoring
  - `scripts/network_utils.py` - Network validation utilities
  - `scripts/fix-iam-permissions.py` - IAM setup
  - `scripts/efs_manager.py` - EFS file system management

- Requirements and dependencies
  - `scripts/requirements.txt` with boto3, python-dotenv, and other dependencies
  - All scripts can be run directly or imported

**Verification**: All deployment scripts tested and operational. Services deploy successfully from scratch and update correctly on existing deployments.

### Stage 0.5: Agent Skip Mode for Connectivity Testing (COMPLETED)

**Goal**: Add special skip mode to agent for easier debugging and connectivity testing.

**Completed Components**:

- Skip phrase detection in `services/agent/app/interface_agent.py`
  - Environment variable `AGENT_SKIP_PHRASE` (default: "skipskipskip")
  - Detects skip phrase in mandate and bypasses full agent execution

- Connectivity testing method (`_test_connectivity()`)
  - Tests RabbitMQ connection status
  - Tests Redis connection (ping)
  - Tests Chroma connection status
  - Tests external API access (HTTP request to external URL)
  - Returns connectivity status dictionary

- Default response with connectivity results
  - Returns structured response with success status
  - Includes connectivity status for each service
  - Clear indication of skip mode usage

**Usage**: Submit task with skip phrase in mandate to test connectivity without running full agent logic.

**Verification**: Skip mode works correctly, connectivity tests accurately reflect service availability.

### Stage 1: Infrastructure Improvements (COMPLETED)

**Goal**: Port stable infrastructure utilities and improve health checks without changing deployment architecture.

**Completed Components**:

- Queue metrics definitions (`services/shared/queue_metrics.py`)
  - QueueDepthMetric dataclass for CloudWatch metric definitions
  - Namespace: "Euglena/RabbitMQ"
  - Metric name: "QueueDepth"
  - Dimension support for queue names
  - Ready for use by metrics publishers (not yet publishing)

- Improved health check configurations (`scripts/build-task-definition.py`)
  - More lenient health check timings to prevent false failures
  - Chroma: start_period=300s, interval=90s, retries=6, timeout=15s
  - Redis: start_period=120s, interval=90s, retries=6, timeout=15s
  - RabbitMQ: start_period=300s, interval=90s, retries=6, timeout=15s
  - Agent: start_period=300s, interval=90s, retries=6, timeout=15s (max allowed by ECS)
  - Gateway: start_period=300s, interval=90s, retries=6, timeout=15s (max allowed by ECS)

- User-Agent header in HTTP connector
  - Already present in `services/agent/app/connector_http.py`
  - Prevents 403 errors from bot detection
  - Uses realistic browser User-Agent string

- Embedding model pre-download
  - Already present in `services/agent/.dockerfile`
  - Pre-downloads all-MiniLM-L6-v2 model during image build
  - Avoids runtime model downloads
  - Note: Increases image size from ~641 MB to ~4.4 GB (disk storage only, not RAM)

- Docker image optimizations
  - Improved layer ordering for better caching
  - Dependencies installed before application code
  - Combined RUN commands to reduce layers
  - .dockerignore file to exclude unnecessary files

**Verification**: No functionality changes, health checks more stable, infrastructure ready for metrics publishing.

## In Progress

### Stage 2: Service Separation with Volume Mounts (EFS WORKING)

**Goal**: Split into gateway and agent services with persistent storage, but keep agent at fixed count (1).

**Completed Components**:

- Task definition split functions in `scripts/build-task-definition.py`
  - `build_gateway_task_definition()` - Creates gateway task with chroma, redis, rabbitmq, gateway containers
  - `build_agent_task_definition()` - Creates agent task with agent container only
  - EFS volume mount support for persistent storage
  - Environment variable transformation for agent to connect to gateway services

**Completed Components**:

- EFS volume configuration (WORKING)
  - IAM authorization enabled for all volumes
  - Transit encryption enabled
  - File system ID: fs-0ec151e2adb754fc8
  - Volumes: chroma-data, redis-data, rabbitmq-data
  - Applied to all task definition functions

- EFS security group access configuration
  - Automatic detection of EFS mount target security groups
  - Automatic addition of NFS (port 2049) ingress rules from ECS task security groups
  - Handles duplicate rules gracefully
  - Runs on every deployment

- EFS mount target management
  - Automatic mount target creation in specified subnets
  - Security group assignment and verification
  - Integration with network discovery

- IAM permissions
  - EFS permissions added to ecsTaskExecutionRole (ClientMount, ClientWrite, ClientRootAccess, DescribeMountTargets)
  - Automatic permission setup during deployment

**Remaining Work**:

- Update `scripts/deploy.py` to build and deploy both task definitions (partially complete - EFS setup integrated)
- Create/update two ECS services (euglena-gateway and euglena-agent)
- Agent environment variable updates for service discovery DNS
- Testing: Deploy both services, verify agent can reach gateway services, verify data persists

**Status**: EFS infrastructure complete and tested. Deployment integration for separate services in progress.

## Planned Stages (Not Yet Started)

### Stage 3: Metrics Service

- Add metrics service to gateway task
- Publish queue depth to CloudWatch
- Use queue_metrics.py definitions from Stage 1

### Stage 4: Lambda Autoscaling Function

- Port Lambda function to read CloudWatch metrics
- Scale agent service based on queue depth
- Fix RabbitMQ tracking issues from autoscale version
- Manual deployment for testing first

### Stage 5: ECS Manager and Task Protection

- Port ECS manager for task protection
- Prevent agents from being killed during work
- Enable/disable task protection on task start/complete

### Stage 6: Service Discovery

- Configure AWS Cloud Map service discovery
- Agent tasks discover gateway via DNS
- Update agent environment variables to use service discovery DNS names

### Stage 7: Full Autoscaling Integration

- Enable full autoscaling with Lambda triggered by EventBridge
- Set conservative limits (MIN_WORKERS=1, MAX_WORKERS=5)
- EventBridge triggers Lambda every 1-2 minutes

### Stage 8: Superdeploy Script

- Create comprehensive deployment script with extended stability testing
- 30-minute stability monitoring
- Health checks every 2 minutes
- Generate stability reports

### Stage 9: Improved Supabase Task Tracking

- Reintroduce StorageManager for Supabase/Redis sync
- Supabase as source of truth for frontend
- Multiple concurrent tasks per user support
- Gateway writes to both Redis and Supabase

## What Currently Works

### Stable Production Features

- Single ECS service deployment with all containers (gateway, agent, redis, rabbitmq, chroma)
- Manual deployment via `scripts/build-task-definition.py`
- Automated deployment via `scripts/deploy.py` (creates or updates services)
- Network auto-discovery (VPC, subnets, security groups)
- ECR image building and pushing with size reporting
- EFS volume support for persistent storage with automatic security group configuration
- Automatic EFS mount target creation and security group rule management
- Health checks with lenient configurations
- Agent skip mode for connectivity testing
- Docker image optimizations for better caching
- User-Agent headers to prevent 403 errors
- Embedding model pre-downloaded in images

### Core Application Features

- Gateway service with Supabase authentication
- Agent worker with dependency injection
- RabbitMQ task queue
- Redis task storage
- ChromaDB for vector storage
- Web interface with real-time status polling
- Task submission and monitoring
- Per-user quota enforcement

## Known Issues and Considerations

### Image Size

- Agent image size increased from ~641 MB to ~4.4 GB due to embedding model pre-download
- This is disk storage only, not RAM usage
- RAM allocation remains 2048 MB (2 GB) for agent tasks (doubled from 1024 MB)
- CPU allocation is 1024 vCPU (doubled from 512 vCPU)
- Cost impact: ~$0.38/month for extra ECR storage
- Trade-off: Faster startup (no model download) vs larger image size

### ECR Push Performance

- Pushing 4.4 GB images to ECR takes significantly longer than smaller images
- Push times can exceed 10+ minutes depending on network speed
- Layer caching helps for subsequent pushes (existing layers skip upload)
- `--skip-ecr` flag available to skip image rebuild/push when only code/config changes
- Consider optimizing image size if push times become problematic:
  - Multi-stage builds to reduce final image size
  - Separate model storage (EFS volume mount) instead of baking into image
  - Compress model files before adding to image

### EFS Configuration

- EFS volumes working and verified
- Transit encryption enabled for all volumes
- IAM authorization enabled for all volumes
- Security group rules automatically configured (NFS port 2049 from ECS task security groups)
- Mount targets automatically created in specified subnets
- IAM permissions configured on ecsTaskExecutionRole (ClientMount, ClientWrite, ClientRootAccess)
- All three volumes (chroma, redis, rabbitmq) use the same EFS file system (fs-0ec151e2adb754fc8)

### Deployment Architecture

- Currently using single task definition with all containers
- Service separation (Stage 2) in progress but not yet deployed
- Autoscaling not yet active (fixed at 1 agent)

### Stability

- Current deployment is stable and production-ready
- No known instability issues with current architecture
- Migration to autoscaling is incremental to maintain stability

## Testing Status

### Verified Working

- Deployment from scratch (fresh AWS account)
- Deployment updates (existing services)
- Network discovery and validation
- ECR image building and pushing
- Health checks (lenient configurations prevent false failures)
- Agent skip mode and connectivity testing
- Docker image layer caching
- EFS volume mounting (WORKING)
- EFS security group configuration
- IAM permissions for EFS access

### Pending Testing

- Service separation (gateway + agent as separate services)
- Agent-to-gateway communication via service discovery
- EFS volume persistence across task restarts (infrastructure ready, needs deployment verification)
- Metrics service and CloudWatch publishing
- Lambda autoscaling function
- Task protection
- Full autoscaling with EventBridge

### Recent Fixes (January 2026)

- Fixed EFS mount access denied errors by configuring security group rules and IAM permissions
- Enabled IAM authorization for all EFS volumes
- Enhanced EFS manager to detect and update mount target security groups automatically
- Added automatic IAM permission setup for EFS access (ecsTaskExecutionRole)
- Improved deployment script to handle duplicate security group rules gracefully
- EFS volumes now working successfully

## Next Steps

1. Complete Stage 2: Finish service separation deployment integration
2. Test service separation: Verify agent can connect to gateway services
3. Begin Stage 3: Add metrics service to gateway task
4. Test metrics publishing to CloudWatch
5. Continue with remaining stages incrementally

## Migration Strategy

The migration follows a conservative staged approach:

- Each stage is tested before proceeding
- 30-minute stability tests at each stage
- Rollback capability at each stage
- No functionality changes in early stages
- Incremental feature addition

This approach helps identify and avoid the instability issues that affected the original autoscale version.

## References

- Full migration plan: `.cursor/plans/autoscale_redux_migration_plan_398a9869.plan.md`
- Architecture documentation: `docs/ARCHITECTURE.md`
- Testing documentation: `docs/TESTING.md`
- ECS service configuration: `docs/ECS_SERVICE_CONFIG.md`

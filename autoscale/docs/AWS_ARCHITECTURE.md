# AWS Architecture and Request Flows

## Overview

This document describes all AWS services used in the Euglena system, their configurations, required permissions, and detailed request flows from ALB to response.

## AWS Services Used

### 1. **Application Load Balancer (ALB)**
- **Purpose**: Routes HTTP/HTTPS traffic to ECS Gateway service
- **Configuration**: 
  - Listens on port 80/443
  - Target group points to Gateway service (port 8080)
  - Health checks: `GET /health` endpoint
- **Access**: Public internet → ALB → Gateway tasks

### 2. **Elastic Container Service (ECS)**
- **Purpose**: Orchestrates containerized services
- **Cluster**: `euglena-cluster`
- **Services**:
  - `euglena-gateway`: Single task with multiple containers (Gateway, Redis, RabbitMQ, ChromaDB)
  - `euglena-agent`: Auto-scaled service (1-11 tasks based on queue depth)
- **Launch Type**: Fargate (serverless containers)
- **Network Mode**: `awsvpc` (each task gets its own ENI)

### 3. **Elastic Container Registry (ECR)**
- **Purpose**: Stores Docker images
- **Repositories**:
  - `euglena/gateway:latest`
  - `euglena/agent:latest`
- **Access**: ECS tasks pull images during deployment

### 4. **Secrets Manager**
- **Purpose**: Stores sensitive configuration
- **Secret ARN**: `arn:aws:secretsmanager:us-east-2:848960888155:secret:euglena/secret-C5JPQd`
- **Secrets Stored**:
  - `OPENAI_API_KEY`
  - `SEARCH_API_KEY`
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY`
  - `SUPABASE_API_KEY`
  - `SUPABASE_JWT_SECRET`
  - `SUPABASE_ALLOW_UNCONFIRMED`
  - `RABBITMQ_ERLANG_COOKIE`
- **Access**: ECS tasks read secrets via task execution role

### 5. **CloudWatch Logs**
- **Purpose**: Centralized logging
- **Log Group**: `/ecs/euglena`
- **Log Streams**:
  - `gateway/{task-id}`
  - `agent/{task-id}`
  - `redis/{task-id}`
  - `rabbitmq/{task-id}`
  - `chroma/{task-id}`
- **Access**: ECS automatically sends container logs

### 6. **CloudWatch Metrics**
- **Purpose**: Queue depth metrics for autoscaling
- **Namespace**: `Euglena/RabbitMQ`
- **Metric**: `QueueDepth`
- **Dimensions**: `QueueName=agent.mandates`
- **Publisher**: Metrics service (separate container in Gateway task)
- **Consumer**: Lambda autoscaling function

### 7. **Lambda**
- **Purpose**: Autoscaling based on queue depth
- **Function**: `lambda-autoscaling` (scheduled execution)
- **Trigger**: EventBridge rule (periodic, e.g., every 1-5 minutes)
- **Actions**:
  - Reads `QueueDepth` metric from CloudWatch
  - Calculates desired worker count
  - Updates ECS service `desiredCount`

### 8. **EventBridge (CloudWatch Events)**
- **Purpose**: Triggers Lambda autoscaling function
- **Rule**: Scheduled rule (e.g., `rate(2 minutes)`)
- **Target**: Lambda function

### 9. **VPC & Service Discovery**
- **Purpose**: Network isolation and service-to-service communication
- **Service Discovery**:
  - Namespace: `euglena.local`
  - Gateway service: `euglena-gateway.euglena.local`
  - Agent tasks resolve this DNS to reach Gateway's Redis/RabbitMQ containers
- **Network**: All tasks in same VPC, security groups allow required ports

### 10. **IAM Roles**

#### **ecsTaskExecutionRole**
- **Used By**: ECS tasks (both Gateway and Agent)
- **Permissions**:
  - `secretsmanager:GetSecretValue` - Read secrets from Secrets Manager
  - `logs:CreateLogStream` - Create CloudWatch log streams
  - `logs:PutLogEvents` - Write logs to CloudWatch
  - `ecr:GetAuthorizationToken` - Pull images from ECR
  - `ecr:BatchCheckLayerAvailability` - Check ECR image layers
  - `ecr:GetDownloadUrlForLayer` - Download ECR image layers
  - `ecr:BatchGetImage` - Get ECR images

#### **ecsTaskRole**
- **Used By**: ECS tasks (both Gateway and Agent)
- **Permissions**:
  - `cloudwatch:PutMetricData` - Publish metrics (for metrics service)
  - `ecs:UpdateService` - Update ECS service (for agent task protection, if implemented)
  - `ecs:DescribeTasks` - Describe ECS tasks (for task metadata)
  - `ecs:StopTask` - Stop tasks (for graceful shutdown, if implemented)

#### **Lambda Execution Role**
- **Used By**: Lambda autoscaling function
- **Permissions**:
  - `cloudwatch:GetMetricStatistics` - Read queue depth metrics
  - `ecs:DescribeServices` - Get current service desired count
  - `ecs:UpdateService` - Update agent service desired count
  - `logs:CreateLogGroup` - Create log groups
  - `logs:CreateLogStream` - Create log streams
  - `logs:PutLogEvents` - Write logs

## Network Architecture

### Gateway Task (Single Task)
```
┌─────────────────────────────────────────┐
│  Gateway Task (awsvpc)                 │
│  ┌──────────┐  ┌──────────┐            │
│  │ Gateway  │  │  Redis   │            │
│  │ :8080    │  │ :6379    │            │
│  └──────────┘  └──────────┘            │
│  ┌──────────┐  ┌──────────┐            │
│  │ RabbitMQ │  │ ChromaDB │            │
│  │ :5672    │  │ :8000    │            │
│  └──────────┘  └──────────┘            │
│  ┌──────────┐                          │
│  │ Metrics  │                          │
│  │ :8082    │                          │
│  └──────────┘                          │
└─────────────────────────────────────────┘
         │
         │ Service Discovery DNS
         │ euglena-gateway.euglena.local
         │
         ▼
┌─────────────────────────────────────────┐
│  Agent Tasks (Multiple, Auto-scaled)    │
│  ┌──────────┐  ┌──────────┐            │
│  │  Agent   │  │  Agent   │  ...        │
│  │ :8081    │  │ :8081    │            │
│  └──────────┘  └──────────┘            │
└─────────────────────────────────────────┘
```

### Port Mappings

**Gateway Task**:
- Gateway container: `8080` (exposed to ALB)
- Redis container: `6379` (exposed to Agent tasks via service discovery)
- RabbitMQ container: `5672` (exposed to Agent tasks via service discovery)
- ChromaDB container: `8000` (exposed to Agent tasks via service discovery)
- Metrics container: `8082` (internal health checks)

**Agent Tasks**:
- Agent container: `8081` (health check endpoint)

### Service Discovery

Agent tasks connect to Gateway services using ECS Service Discovery:
- **Redis**: `redis://euglena-gateway.euglena.local:6379/0`
- **RabbitMQ**: `amqp://guest:guest@euglena-gateway.euglena.local:5672/`
- **ChromaDB**: `http://euglena-gateway.euglena.local:8000`

ECS automatically creates DNS records that resolve to the Gateway task's private IP.

## Request Flows

### Flow 1: Submit Task (`POST /tasks`)

**Request Path**: `Client → ALB → Gateway Container → Redis → RabbitMQ → Agent Tasks`

```
1. Client sends POST /tasks with:
   - Authorization: Bearer <Supabase JWT>
   - Body: { "mandate": "...", "max_ticks": 50 }

2. ALB receives request
   - Routes to Gateway service target group
   - Health check passes → forwards to Gateway container :8080

3. Gateway Container (FastAPI)
   a. CORS middleware validates origin
   b. HTTPBearer extracts JWT token
   c. get_current_supabase_user() validates JWT:
      - Decodes JWT using SUPABASE_JWT_SECRET (from Secrets Manager)
      - Validates signature, expiration, issuer
      - Returns SupabaseUser object
   
   d. Quota check (if not test mode):
      - SupabaseUserTickManager.check_and_consume()
      - Queries Supabase for user's daily tick usage
      - Compares against DAILY_TICK_LIMIT (256)
      - If exceeded → 429 Too Many Requests
   
   e. GatewayService.create_task():
      - Generates correlation_id (UUID)
      - Creates TaskRecord with status="pending"
      - Stores in Redis (localhost:6379):
        * Key: task:{correlation_id}
        * Value: JSON task data
      - Verifies storage by reading back
   
   f. Publishes to RabbitMQ (localhost:5672):
      - Queue: agent.mandates
      - Payload: TaskEnvelope (mandate, max_ticks, correlation_id)
   
   g. Returns TaskResponse:
      - Status: 202 Accepted
      - Body: { correlation_id, status: "in_queue", ... }

4. Response flows back:
   Gateway → ALB → Client
```

**AWS Services Involved**:
- ALB: Request routing
- ECS: Gateway task execution
- Secrets Manager: JWT secret retrieval
- CloudWatch Logs: Request logging
- Redis (sidecar): Task storage
- RabbitMQ (sidecar): Message queue

**IAM Permissions Required**:
- `ecsTaskExecutionRole`: Read secrets, write logs
- No additional permissions needed (Redis/RabbitMQ are localhost)

---

### Flow 2: Get Task Status (`GET /tasks/{correlation_id}`)

**Request Path**: `Client → ALB → Gateway Container → Redis`

```
1. Client sends GET /tasks/{correlation_id} with:
   - Authorization: Bearer <Supabase JWT>

2. ALB routes to Gateway container :8080

3. Gateway Container:
   a. Validates JWT (same as Flow 1)
   b. GatewayService.get_task(correlation_id):
      - Reads from Redis (localhost:6379):
        * Key: task:{correlation_id}
        * Returns JSON task data
      - If not found → 404 Not Found
      - Normalizes status:
        * "pending" → "in_queue"
        * "accepted"/"in_progress" → "in_progress"
        * Others unchanged
   
   c. Returns TaskResponse:
      - Status: 200 OK
      - Body: { correlation_id, status, mandate, result, error, tick, ... }

4. Response flows back:
   Gateway → ALB → Client
```

**AWS Services Involved**:
- ALB: Request routing
- ECS: Gateway task execution
- Secrets Manager: JWT secret retrieval
- CloudWatch Logs: Request logging
- Redis (sidecar): Task storage

**IAM Permissions Required**:
- `ecsTaskExecutionRole`: Read secrets, write logs

---

### Flow 3: Get Agent Count (`GET /agents/count`)

**Request Path**: `Client → ALB → Gateway Container → Redis`

```
1. Client sends GET /agents/count
   - No authentication required (public endpoint)

2. ALB routes to Gateway container :8080

3. Gateway Container:
   a. GatewayService.get_agent_count():
      - Connects to Redis (localhost:6379)
      - Reads Redis set: workers:status
      - Counts members: SCARD workers:status
      - Returns count (0 on error)
   
   b. Returns:
      - Status: 200 OK
      - Body: { "count": 5 }

4. Response flows back:
   Gateway → ALB → Client
```

**How Agent Count Works**:
- Each Agent task registers itself in Redis when it starts
- WorkerPresence class maintains heartbeat:
  - Adds worker_id to set: `workers:status`
  - Sets TTL key: `worker:agent:{worker_id}` (expires after 30s)
  - Refreshes every 10 seconds (AGENT_STATUS_TIME)
- Gateway counts active workers from the set
- Stale workers removed after TTL expires

**AWS Services Involved**:
- ALB: Request routing
- ECS: Gateway and Agent task execution
- CloudWatch Logs: Request logging
- Redis (sidecar): Worker presence storage

**IAM Permissions Required**:
- `ecsTaskExecutionRole`: Write logs

---

### Flow 4: Task Processing (Agent Consumes and Executes)

**Flow Path**: `RabbitMQ → Agent Task → Redis → ChromaDB → External APIs`

```
1. Agent Task Startup:
   a. ECS starts Agent task (from autoscaling or initial deployment)
   b. Agent container initializes:
      - Connects to RabbitMQ: euglena-gateway.euglena.local:5672
      - Connects to Redis: euglena-gateway.euglena.local:6379
      - Connects to ChromaDB: euglena-gateway.euglena.local:8000
      - Initializes connectors (LLM, Search, HTTP)
   
   c. WorkerPresence.run():
      - Registers in Redis: SADD workers:status {worker_id}
      - Starts heartbeat loop (every 10s)

2. Agent Consumes Task:
   a. InterfaceAgent._start_consumer():
      - Consumes from queue: agent.mandates
      - Receives TaskEnvelope: { mandate, max_ticks, correlation_id }
   
   b. _handle_task():
      - Updates Redis: status="accepted"
      - Enables ECS task protection (if ECS_ENABLED)
      - Updates Redis: status="in_progress"
      - Starts heartbeat loop (updates tick every 10s)

3. Agent Execution:
   a. Creates Agent instance with mandate
   b. Agent.run() - Tick-based loop:
      - Tick 1: Build prompt, call LLM, parse response
      - Execute action (SEARCH, VISIT, THINK, EXIT)
      - Store context in ChromaDB
      - Retrieve relevant context from ChromaDB
      - Repeat until max_ticks or EXIT action
   
   c. External API calls:
      - LLM: OpenAI API (MODEL_API_URL)
      - Search: Search API (SEARCH_API_KEY)
      - HTTP: Web requests for URL visits
   
   d. Status updates (every 10s):
      - Updates Redis: { status: "in_progress", tick: N }

4. Task Completion:
   a. Agent.run() returns result
   b. Updates Redis: status="completed", result={...}
   c. Disables ECS task protection
   d. Updates worker status: FREE
   e. Agent ready for next task

5. Gateway Status Polling:
   - Client polls GET /tasks/{correlation_id}
   - Gateway reads from Redis
   - Returns latest status, tick, result
```

**AWS Services Involved**:
- ECS: Agent task execution, autoscaling
- Service Discovery: DNS resolution for Gateway services
- Secrets Manager: API keys (OPENAI_API_KEY, SEARCH_API_KEY)
- CloudWatch Logs: Agent execution logs
- CloudWatch Metrics: Queue depth (published by metrics service)
- Redis (Gateway sidecar): Task status storage
- RabbitMQ (Gateway sidecar): Task queue
- ChromaDB (Gateway sidecar): Vector storage

**IAM Permissions Required**:
- `ecsTaskExecutionRole`: Read secrets, write logs
- `ecsTaskRole`: Publish metrics (if metrics service), update task protection

---

### Flow 5: Autoscaling (Lambda → CloudWatch → ECS)

**Flow Path**: `EventBridge → Lambda → CloudWatch → ECS`

```
1. EventBridge Rule triggers:
   - Scheduled: rate(2 minutes) or cron expression
   - Invokes Lambda function: lambda-autoscaling

2. Lambda Function Execution:
   a. Reads configuration:
      - ECS_CLUSTER: euglena-cluster
      - ECS_SERVICE: euglena-agent
      - QUEUE_NAME: agent.mandates
      - TARGET_MESSAGES_PER_WORKER: 1
      - MIN_WORKERS: 1
      - MAX_WORKERS: 11
   
   b. get_queue_depth():
      - Calls CloudWatch: GetMetricStatistics
      - Namespace: Euglena/RabbitMQ
      - Metric: QueueDepth
      - Dimensions: QueueName=agent.mandates
      - Period: Last 5 minutes, average
      - Returns: Queue depth (e.g., 5)
   
   c. get_current_worker_count():
      - Calls ECS: DescribeServices
      - Returns: Current desiredCount (e.g., 3)
   
   d. calculate_desired_workers():
      - If queue_depth == 0: return MIN_WORKERS (1)
      - Otherwise: ceil(queue_depth / TARGET_MESSAGES_PER_WORKER)
      - Cap at MAX_WORKERS (11)
      - Example: ceil(5 / 1) = 5 workers
   
   e. update_service_desired_count():
      - If desired != current:
        * Calls ECS: UpdateService
        * Sets desiredCount = 5
        * ECS automatically starts/stops tasks

3. ECS Autoscaling:
   a. ECS receives UpdateService request
   b. If desiredCount > current:
      - Starts new Agent tasks
      - Tasks register in Redis (workers:status)
      - Tasks consume from RabbitMQ queue
   
   c. If desiredCount < current:
      - Stops excess tasks (graceful shutdown)
      - Tasks remove themselves from Redis
      - Tasks finish current work before stopping

4. Metrics Service (Continuous):
   - Runs in Gateway task (separate container)
   - Every 10 seconds:
     * Reads queue depth from RabbitMQ
     * Publishes to CloudWatch: PutMetricData
     * Namespace: Euglena/RabbitMQ
     * Metric: QueueDepth
```

**AWS Services Involved**:
- EventBridge: Scheduled trigger
- Lambda: Autoscaling logic
- CloudWatch Metrics: Queue depth storage
- CloudWatch Logs: Lambda execution logs
- ECS: Service scaling
- RabbitMQ (Gateway sidecar): Queue depth source
- Redis (Gateway sidecar): Worker registration

**IAM Permissions Required**:
- Lambda execution role:
  - `cloudwatch:GetMetricStatistics`
  - `ecs:DescribeServices`
  - `ecs:UpdateService`
  - `logs:CreateLogStream`, `logs:PutLogEvents`
- Metrics service (ecsTaskRole):
  - `cloudwatch:PutMetricData`

## Security Groups

### Gateway Task Security Group
**Inbound Rules**:
- Port 8080: From ALB security group (HTTP/HTTPS)
- Port 6379: From Agent task security group (Redis)
- Port 5672: From Agent task security group (RabbitMQ)
- Port 8000: From Agent task security group (ChromaDB)

**Outbound Rules**:
- All traffic (for external API calls: OpenAI, Search, HTTP)

### Agent Task Security Group
**Inbound Rules**:
- Port 8081: From Gateway task security group (health checks, optional)

**Outbound Rules**:
- Port 6379: To Gateway task security group (Redis)
- Port 5672: To Gateway task security group (RabbitMQ)
- Port 8000: To Gateway task security group (ChromaDB)
- HTTPS: To external APIs (OpenAI, Search, web)

### ALB Security Group
**Inbound Rules**:
- Port 80: From 0.0.0.0/0 (HTTP)
- Port 443: From 0.0.0.0/0 (HTTPS)

**Outbound Rules**:
- Port 8080: To Gateway task security group

## Summary

### Request Flow Summary

1. **Submit Task**: Client → ALB → Gateway → Redis (store) → RabbitMQ (publish) → 202 Accepted
2. **Get Status**: Client → ALB → Gateway → Redis (read) → 200 OK with status
3. **Agent Count**: Client → ALB → Gateway → Redis (count workers set) → 200 OK with count
4. **Task Processing**: RabbitMQ → Agent → Redis (status updates) → ChromaDB (context) → External APIs → Redis (completion)
5. **Autoscaling**: EventBridge → Lambda → CloudWatch (read metrics) → ECS (update service) → ECS (scale tasks)

### Key AWS Features

- **Service Discovery**: Enables Agent tasks to find Gateway services via DNS
- **Secrets Manager**: Secure storage of API keys and configuration
- **CloudWatch Metrics**: Queue depth tracking for autoscaling
- **Lambda Autoscaling**: Dynamic worker scaling based on queue depth
- **Fargate**: Serverless container execution (no EC2 management)
- **ALB**: Public-facing load balancer with health checks
- **VPC Networking**: Isolated network with security groups

### Access Requirements Summary

**Gateway Task**:
- Read secrets (JWT, API keys)
- Write logs
- Publish metrics (metrics container)
- Access localhost services (Redis, RabbitMQ, ChromaDB)

**Agent Task**:
- Read secrets (API keys)
- Write logs
- Connect to Gateway services via service discovery
- Access external APIs (OpenAI, Search, web)

**Lambda Function**:
- Read CloudWatch metrics
- Read ECS service status
- Update ECS service desired count
- Write logs

**ALB**:
- Route HTTP/HTTPS traffic
- Health check Gateway service
- No AWS API access needed

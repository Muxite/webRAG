# ECS Deployment Settings

Configuration guide for deploying Gateway and Agent services on AWS ECS with Lambda autoscaling.

## Prerequisites

**Only Network Infrastructure Required:**
- VPC with public and private subnets (at least 2 AZs)
- Internet Gateway attached to VPC
- NAT Gateway in public subnet (for private subnet internet access)
- Security Groups configured (see Network Architecture section)
- Route tables configured

**Everything Else is Automated:**
- The `build-task-definition.py` script automatically:
  - Loads secrets from `keys.env` and adds them to task definitions via Secrets Manager
  - Loads environment variables from `.env` and adds them to task definitions
  - Configures service discovery URLs for agent service
  - Sets up all container definitions with proper configuration

**Other AWS Resources (created via scripts/console):**
- IAM Roles (ecsTaskRole, ecsTaskExecutionRole, Lambda role)
- ECR repositories (created by push-to-ecr.py script)
- Secrets Manager secret (created by create-secrets.py script)
- ECS Cluster
- Cloud Map namespace (for service discovery)

---

## 1. Lambda Autoscaling Deployment

### Lambda Function Configuration

**Basic Settings:**
- **Function name:** `euglena-autoscaling`
- **Runtime:** Python 3.10 or 3.11
- **Architecture:** x86_64
- **Handler:** `lambda_function.lambda_handler`
- **Timeout:** 60 seconds
- **Memory:** 128 MB (minimum, can increase if needed)

**Deployment Package:**

Create a ZIP file with the following structure:
```
deployment-package.zip
├── lambda_function.py          # The main Lambda function code
├── aws.env                     # AWS configuration (from services/aws.env)
├── .env                        # Environment variables (from services/.env)
└── [dependencies]              # Installed Python packages (boto3, python-dotenv, etc.)
```

**Steps to create:**
1. Copy `services/lambda_autoscaling/lambda_function.py` to a temp directory
2. Copy `services/aws.env` and `services/.env` to the same directory
3. Install dependencies: `pip install -r services/lambda_autoscaling/requirements.txt -t .`
4. Zip everything: `zip -r deployment-package.zip .`
5. Upload to Lambda

**Package size limits:**
- Max 50MB uncompressed
- Max 250MB unzipped
- Can use Lambda layers for large dependencies

**Note:** The code looks for `aws.env` and `.env` in the same directory as `lambda_function.py` (using `Path(__file__).parent.parent`, which resolves to the package root in Lambda).

**IAM Role Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricStatistics"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeServices",
        "ecs:UpdateService"
      ],
      "Resource": "arn:aws:ecs:us-east-2:848960888155:service/euglena-cluster/euglena-agent"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

Replace ARN values with actual region, account ID, cluster name, and service name.

ARN format: `arn:aws:ecs:REGION:ACCOUNT_ID:service/CLUSTER_NAME/SERVICE_NAME`

VPC Configuration: Not required. Lambda reads queue depth from CloudWatch Metrics, not directly from RabbitMQ. CloudWatch and ECS APIs are public endpoints. Queue depth metrics must be published to CloudWatch by metrics service.

**Environment Variables:**
- None needed (loads from deployment package's aws.env and .env)

**EventBridge Trigger:**
- **Rule name:** `euglena-autoscaling-trigger`
- **Schedule:** Rate expression: `rate(2 minutes)` or Cron: `cron(*/2 * * * ? *)`
- **Target:** Lambda function `euglena-autoscaling`

---

## 2. ECS Gateway Service Deployment

### Task Definition Settings

**Basic Configuration:**
- **Family:** `euglena-gateway`
- **Launch type:** Fargate
- **Network mode:** `awsvpc`
- **CPU:** 512 (0.5 vCPU)
- **Memory:** 1024 MB (1 GB)
- **Task role ARN:** `arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole`
- **Execution role ARN:** `arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole`

**Container Definitions:**

1. **Gateway Container:**
   - **Name:** `gateway`
   - **Image:** `ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/euglena/gateway:latest`
   - **Port:** 8080
   - **Health check:** `CMD-SHELL curl -f http://localhost:8080/health || exit 1`
   - **Start period:** 300 seconds
   - **Interval:** 60 seconds
   - **Timeout:** 10 seconds
   - **Retries:** 5

2. **Redis Container:**
   - **Name:** `redis`
   - **Image:** `redis:7-alpine`
   - **Port:** 6379
   - **Health check:** `CMD-SHELL redis-cli ping | grep PONG || exit 1`
   - **Start period:** 90 seconds

3. **RabbitMQ Container:**
   - **Name:** `rabbitmq`
   - **Image:** `rabbitmq:3-management`
   - **Ports:** 5672, 15672
   - **Health check:** `CMD-SHELL rabbitmq-diagnostics ping || exit 1`
   - **Start period:** 180 seconds

4. **Chroma Container:**
   - **Name:** `chroma`
   - **Image:** `chromadb/chroma:latest`
   - **Port:** 8000
   - **Health check:** `CMD-SHELL curl -f http://localhost:8000/api/v1/heartbeat || exit 1`
   - **Start period:** 180 seconds
   - **Environment:** `IS_PERSISTENT=TRUE`, `PERSIST_DIRECTORY=/chroma-data`

**Secrets (from Secrets Manager):**
- All secrets from `keys.env` (OPENAI_API_KEY, SEARCH_API_KEY, SUPABASE_JWT_SECRET, etc.)
- Automatically added by build script
- Format: `arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:SECRET_NAME-SUFFIX:KEY_NAME::`

**Environment Variables:**
- Automatically loaded from `.env` file by build script
- Gateway uses container names (redis, chroma, rabbitmq) for service URLs
- Includes all variables from `.env` (excludes secrets and lambda-specific vars)

### ECS Service Configuration

**Service Settings:**
- **Service name:** `euglena-gateway`
- **Cluster:** `euglena-cluster`
- **Task definition:** `euglena-gateway` (latest revision)
- **Desired count:** 2 (minimum for high availability)
- **Launch type:** Fargate
- **Platform version:** Latest (1.4.0 or newer)

**Network Configuration:**
- **VPC:** Your VPC ID
- **Subnets:** 
  - **Public subnets** (for ALB access)
  - At least 2 subnets in different AZs
- **Security groups:** 
  - Dedicated gateway security group
  - **Inbound:** Port 8080 from ALB security group, ports 6379/5672/8000 from agent security group
  - **Outbound:** All traffic (0.0.0.0/0)
- **Auto-assign public IP:** ENABLED (required for public subnets)

**Load Balancing:**
- **Load balancer type:** Application Load Balancer (ALB)
- **Target group:** 
  - **Name:** `euglena-gateway-tg`
  - **Protocol:** HTTP
  - **Port:** 8080
  - **Health check path:** `/health`
  - **Health check interval:** 30 seconds
  - **Healthy threshold:** 2
  - **Unhealthy threshold:** 3
  - **Timeout:** 5 seconds
- **Container to load balance:** `gateway:8080`

**Service Discovery:**
- **Namespace:** `euglena.local`
- **Service name:** `euglena-gateway`
- **DNS name:** `euglena-gateway.euglena.local`

**Auto Scaling:**
- **Min capacity:** 2
- **Max capacity:** 5-10
- **Target tracking:** Request count per target (500-1000 requests per target)

**Deployment Configuration:**
- **Deployment type:** Rolling update
- **Minimum healthy percent:** 100%
- **Maximum percent:** 200%
- **Deployment circuit breaker:** Enabled (rollback on failure)

### IAM Roles

**Task Role (`ecsTaskRole`):**
- Allows task to access AWS services (S3, Secrets Manager, etc.)
- Permissions depend on what your application needs

**Execution Role (`ecsTaskExecutionRole`):**
- Allows ECS to pull images and secrets
- Required permissions:
  - `ecr:GetAuthorizationToken`
  - `ecr:BatchCheckLayerAvailability`
  - `ecr:GetDownloadUrlForLayer`
  - `ecr:BatchGetImage`
  - `secretsmanager:GetSecretValue`
  - `logs:CreateLogStream`
  - `logs:PutLogEvents`

---

## 3. ECS Agent Service Deployment

### Task Definition Settings

**Basic Configuration:**
- **Family:** `euglena-agent`
- **Launch type:** Fargate
- **Network mode:** `awsvpc`
- **CPU:** 256 (0.25 vCPU)
- **Memory:** 2048 MB (2 GB)
- **Task role ARN:** `arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole`
- **Execution role ARN:** `arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole`

**Container Definitions:**

1. **Agent Container:**
   - **Name:** `agent`
   - **Image:** `ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/euglena/agent:latest`
   - **Port:** 8081
   - **Health check:** `CMD-SHELL curl -f http://localhost:8081/health || exit 1`
   - **Start period:** 60 seconds
   - **Interval:** 60 seconds
   - **Timeout:** 10 seconds
   - **Retries:** 5

**Secrets (from Secrets Manager):**
- All secrets from `keys.env` (OPENAI_API_KEY, SEARCH_API_KEY, etc.)
- Automatically added by build script

**Environment Variables:**
- Automatically loaded from `.env` file by build script
- Service URLs configured for service discovery:
  - `REDIS_URL=redis://euglena-gateway.euglena.local:6379/0`
  - `CHROMA_URL=http://euglena-gateway.euglena.local:8000`
  - `RABBITMQ_URL=amqp://guest:guest@euglena-gateway.euglena.local:5672/`
- ECS-specific vars added:
  - `ECS_ENABLED=true`
  - `AWS_REGION` (from aws.env)
  - `ECS_CLUSTER` (from aws.env)
- Includes all other variables from `.env`

### ECS Service Configuration

**Service Settings:**
- **Service name:** `euglena-agent`
- **Cluster:** `euglena-cluster`
- **Task definition:** `euglena-agent` (latest revision)
- **Desired count:** 1 (minimum, managed by Lambda autoscaling)
- **Launch type:** Fargate
- **Platform version:** Latest

**Network Configuration:**
- **VPC:** Same VPC as gateway
- **Subnets:** 
  - Private subnets only
  - At least 2 subnets in different AZs
- **Security groups:**
  - **Inbound:** None (agents don't receive external traffic)
  - **Outbound:** All traffic (0.0.0.0/0) for API calls
- **Auto-assign public IP:** DISABLED (use NAT Gateway for internet)

**Service Discovery:**
- **Namespace:** `euglena.local`
- **Service name:** `euglena-agent`

**Auto Scaling:**
- Managed by Lambda autoscaling function
- Min capacity: 1 (from MIN_WORKERS in .env)
- Max capacity: 11 (from MAX_WORKERS in .env)
- Do not enable ECS auto-scaling (conflicts with Lambda)

**Deployment Configuration:**
- **Deployment type:** Rolling update
- **Minimum healthy percent:** 0% (allows scaling to zero)
- **Maximum percent:** 200%
- **Deployment circuit breaker:** Enabled

---

## 4. Network Architecture

### VPC Structure

```
VPC (10.0.0.0/16)
├── Public Subnet 1 (10.0.1.0/24) - AZ us-east-2a
│   ├── NAT Gateway
│   └── Internet Gateway (attached to VPC)
├── Public Subnet 2 (10.0.2.0/24) - AZ us-east-2b
│   └── NAT Gateway (optional, for HA)
├── Private Subnet 1 (10.0.11.0/24) - AZ us-east-2a
│   └── Agent tasks
└── Private Subnet 2 (10.0.12.0/24) - AZ us-east-2b
    └── Agent tasks
```

### Security Groups

**ALB Security Group:**
- **Inbound:** 
  - Port 80/443 from 0.0.0.0/0 (or specific IPs)
- **Outbound:**
  - Port 8080 to Gateway Security Group

**Gateway Security Group:**
- **Inbound:**
  - Port 8080 from ALB Security Group
  - Port 6379 from Agent Security Group (Redis)
  - Port 5672 from Agent Security Group (RabbitMQ)
  - Port 8000 from Agent Security Group (Chroma)
- **Outbound:**
  - All traffic (0.0.0.0/0)

**Agent Security Group:**
- **Inbound:**
  - None (agents don't receive external traffic)
- **Outbound:**
  - All traffic (0.0.0.0/0) for API calls, service discovery

### Service Discovery

**Cloud Map Namespace:**
- **Namespace:** `euglena.local`
- **Type:** Private DNS namespace
- **VPC:** Your VPC

**Service Discovery Services:**
- **Gateway:** `euglena-gateway.euglena.local`
  - Resolves to gateway task IPs
  - Port 8080 (HTTP), 6379 (Redis), 5672 (RabbitMQ), 8000 (Chroma)
- **Agent:** `euglena-agent.euglena.local` (optional)
  - Resolves to agent task IPs

---

## 5. IAM Roles Summary

### ECS Task Execution Role

**Name:** `ecsTaskExecutionRole`

**Trust Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Permissions:**
- ECR image pull
- Secrets Manager access
- CloudWatch Logs

### ECS Task Role

**Name:** `ecsTaskRole`

**Trust Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Permissions:**
- Custom permissions based on application needs
- May include: S3 access, DynamoDB, etc.

---

## 6. Deployment Checklist

### Pre-Deployment
**Network Setup (Required):**
- [ ] VPC and subnets created (public and private, 2+ AZs)
- [ ] Internet Gateway attached to VPC
- [ ] NAT Gateway in public subnet
- [ ] Security groups created (see Network Architecture section)
- [ ] Route tables configured

**AWS Resources:**
- [ ] IAM roles created (ecsTaskRole, ecsTaskExecutionRole, Lambda role)
- [ ] ECR repositories created
- [ ] Secrets Manager secret created (use `create-secrets.py` script)
- [ ] ECS cluster created
- [ ] Cloud Map namespace created (for service discovery)

### Lambda
- [ ] Lambda function created
- [ ] IAM role attached
- [ ] Deployment package uploaded
- [ ] EventBridge rule created
- [ ] Test invocation works

### Gateway Service
- [ ] Images pushed to ECR (use `push-to-ecr.py` script)
- [ ] Run `build-task-definition.py` script (automatically includes secrets and env vars)
- [ ] Register task definition from generated JSON
- [ ] Service created with task definition
- [ ] ALB and target group created
- [ ] Service discovery configured
- [ ] Health checks passing

### Agent Service
- [ ] Images pushed to ECR (use `push-to-ecr.py` script)
- [ ] Run `build-task-definition.py` script (automatically includes secrets, env vars, and service discovery URLs)
- [ ] Register task definition from generated JSON
- [ ] Service created with task definition
- [ ] Lambda autoscaling configured
- [ ] Test scaling works

---

## 7. Environment Variables Reference

The `build-task-definition.py` script automatically loads environment variables from `.env`, excludes secrets handled via Secrets Manager, and configures service URLs appropriately. Ensure `.env` has required variables.

### Gateway Service Environment Variables

**Service URLs (via service discovery or direct):**
- `REDIS_URL=redis://redis:6379/0` (container name in same task)
- `CHROMA_URL=http://chroma:8000` (container name in same task)
- `RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/` (container name in same task)

**LLM Configuration:**
- `MODEL_API_URL=https://api.openai.com/v1/`
- `MODEL_NAME=gpt-4o`

**Supabase:**
- `SUPABASE_URL=...`
- `SUPABASE_ANON_KEY=...`
- `SUPABASE_JWT_SECRET=...` (from Secrets Manager)

**CORS:**
- `CORS_ALLOWED_ORIGINS=...`

### Agent Service Environment Variables

**Service URLs (via service discovery):**
- `REDIS_URL=redis://euglena-gateway.euglena.local:6379/0`
- `CHROMA_URL=http://euglena-gateway.euglena.local:8000`
- `RABBITMQ_URL=amqp://guest:guest@euglena-gateway.euglena.local:5672/`

**LLM Configuration:**
- `MODEL_API_URL=https://api.openai.com/v1/`
- `MODEL_NAME=gpt-4o`

**ECS Configuration:**
- `ECS_ENABLED=true`
- `AWS_REGION=us-east-2`
- `ECS_CLUSTER=euglena-cluster`

**Queue Configuration:**
- `AGENT_INPUT_QUEUE=agent.mandates`

**Secrets (from Secrets Manager):**
- `OPENAI_API_KEY`
- `SEARCH_API_KEY`

---

## 8. Common Issues

### Network Issues
- **Agents can't reach gateway:** Check security groups, service discovery DNS
- **No internet access:** Verify NAT Gateway, route tables
- **Health checks failing:** Check security group rules, health check path

### Scaling Issues
- **Lambda not scaling:** Check CloudWatch metrics, IAM permissions
- **Service stuck:** Check deployment configuration, task failures

### Service Discovery Issues
- **DNS not resolving:** Verify namespace, service registration
- **Wrong IPs:** Check service discovery health checks

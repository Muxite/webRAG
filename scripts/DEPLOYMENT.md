# Deployment Scripts

Scripts for AWS ECS deployment and management.

## Scripts

- **`deploy.py`**: Full deployment automation (ECR push, task definitions, ECS service)
- **`build-task-definition.py`**: Generate and register ECS task definitions from env files
- **`register-secrets.py`**: Register/update secrets in AWS Secrets Manager
- **`network_discovery.py`**: Auto-discover VPC, subnets, security groups (OOP module)
- **`ecs_infrastructure.py`**: Manage ECS clusters and services (OOP module)
- **`network_utils.py`**: Network validation and security group fixes
- **`fix-iam-permissions.py`**: Setup IAM permissions for ECS task protection
- **`check.py`**: Health checks for deployed services

## Quick Start

All scripts must be run from the `services/` directory.

### Full Deployment

```bash
cd services
python ../scripts/deploy.py
```

Deploys everything:
- Builds and pushes Docker images to ECR
- Generates and registers task definitions
- Creates/updates ECS service with ALB integration
- Configures IAM permissions

Options:
- `--skip-ecr`: Skip Docker build/push
- `--skip-network-check`: Skip network validation
- `--wait`: Wait for service to stabilize after deployment

### Register Secrets

```bash
cd services
python ../scripts/register-secrets.py
```

Automatically creates or updates the secret in AWS Secrets Manager.

### Generate Task Definitions

```bash
cd services
python ../scripts/build-task-definition.py
```

Generates task definition JSON from `keys.env` and `.env` files.

### Comprehensive Diagnostics

```bash
cd services
python ../scripts/diagnose.py [--service SERVICE_NAME] [--hours HOURS]
```

Gathers extensive diagnostic data including:
- Service and task status
- Container health and logs (recent and error logs)
- CloudWatch metrics
- EFS mount status
- Task definition details
- Resource utilization

Options:
- `--service`: Service name (default: from aws.env ECS_SERVICE_NAME)
- `--hours`: Hours of logs to retrieve (default: 24)

## Configuration

Scripts read from `services/aws.env`:
- AWS account ID, region, cluster name
- VPC, subnet, security group IDs
- ALB and target group configuration
- Secret names and ARN suffixes

Keys are loaded from `services/keys.env` for secret registration.

## Requirements

Install dependencies:

```bash
pip install -r scripts/requirements.txt
```

## Service Configuration

The deployment creates a single ECS service `euglena-service` with:
- Task definition: `euglena` (all containers in one task)
- Desired count: 1
- Capacity provider: FARGATE
- ALB integration: Routes to `gateway` container on port 8080
- Health check grace period: 100 seconds

See `docs/ECS_SERVICE_CONFIG.md` for complete configuration details.

# Deployment Scripts

Scripts for AWS ECS deployment and management.

## Scripts

- **`deploy-single.py`**: Deploy single service mode (all containers in one service)
- **`deploy-autoscale.py`**: Deploy autoscale mode (separate gateway and agent services)
- **`build-task-definition.py`**: Generate and register ECS task definitions from env files
- **`register-secrets.py`**: Register/update secrets in AWS Secrets Manager
- **`network_discovery.py`**: Auto-discover VPC, subnets, security groups (OOP module)
- **`ecs_infrastructure.py`**: Manage ECS clusters and services (OOP module)
- **`network_utils.py`**: Network validation and security group fixes
- **`fix-iam-permissions.py`**: Setup IAM permissions for ECS task protection
- **`check.py`**: Health checks for deployed services
- **`diagnose-deployment.py`**: Comprehensive deployment diagnostics
- **`efs-tools.py`**: EFS verification and diagnostic tools
- **`verify-config.py`**: Verify task definition and security group configuration

## Quick Start

All scripts must be run from the `services/` directory.

### Full Deployment

**Single Service Mode:**
```bash
cd services
python ../scripts/deploy-single.py [--skip-ecr] [--skip-network-check] [--wait]
```

**Autoscale Mode:**
```bash
cd services
python ../scripts/deploy-autoscale.py [--skip-ecr] [--skip-network-check] [--wait]
```

Deploys everything:
- Builds and pushes Docker images to ECR
- Generates and registers task definitions
- Creates/updates ECS service(s) with ALB integration
- Configures IAM permissions
- Sets up service discovery (autoscale mode only)

Options:
- `--skip-ecr`: Skip Docker build/push
- `--skip-network-check`: Skip network validation
- `--wait`: Wait for service(s) to stabilize after deployment

### Register Secrets

```bash
cd services
python ../scripts/register-secrets.py
```

Automatically creates or updates the secret in AWS Secrets Manager.

### Verify Configuration

```bash
cd services
python ../scripts/verify-config.py [--mode task-def|security-groups|all]
```

Validates task definition and security group configuration:
- Task definition: EFS volume configuration, mount points, IAM permissions
- Security groups: ECS service security groups, EFS mount target security groups, NFS access rules

Options:
- `--mode`: Verification mode - task-def, security-groups, or all (default: all)
- `--task-family FAMILY`: Task definition family name (default: from aws.env)
- `--file PATH`: Path to task definition JSON file
- `--from-aws`: Get task definition from AWS instead of local file

### Health Checks

```bash
cd services
python ../scripts/check.py [--mode single|autoscale] [--service gateway|agent|all] [--verbose]
```

Check health of deployed services.

Options:
- `--mode`: Deployment mode - single or autoscale (default: single)
- `--service`: Service to check - gateway, agent, or all (default: all, only for autoscale mode)
- `--verbose`: Show verbose output

### EFS Tools

```bash
cd services
python ../scripts/efs-tools.py [--mode verify|diagnose]
```

EFS verification and diagnostic tools:
- Verify mode: EFS filesystem accessibility, security groups, mount targets, routing, IAM permissions
- Diagnose mode: Detailed file system info, mount targets, access points, policy, recent task failures

Options:
- `--mode`: verify (default) or diagnose
- `--file-system-id ID`: Override EFS file system ID from aws.env
- `--hours N`: Hours to look back for ECS task errors (diagnose mode, default: 24)

### Generate Task Definitions

```bash
cd services
python ../scripts/build-task-definition.py [--mode single|autoscale]
```

Generates task definition JSON from `keys.env` and `.env` files.

Modes:
- `--mode single` (default): Single task definition with all containers
- `--mode autoscale`: Autoscale task definitions for gateway and agent

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

### Single Service Mode

Creates a single ECS service `euglena-service` with:
- Task definition: `euglena` (all containers in one task)
- Resources: 1 vCPU (1024 CPU units), 2GB RAM (2048 MB)
- Desired count: 1
- Capacity provider: FARGATE
- ALB integration: Routes to `gateway` container on port 8080
- Health check grace period: 100 seconds
- Chroma is optional - agent continues working even if Chroma health checks fail

### Autoscale Service Mode

Creates two ECS services:
- `euglena-gateway`: Gateway service with ALB integration
  - Resources: 0.5 vCPU (512 CPU units), 1GB RAM (1024 MB)
  - Containers: chroma, redis, rabbitmq, gateway
  - Service discovery: Registered at `euglena-gateway.euglena.local`
- `euglena-agent`: Agent service (no ALB)
  - Resources: 0.25 vCPU (256 CPU units), 0.5GB RAM (512 MB)
  - Containers: agent
  - Connects to gateway via service discovery DNS

Both services use FARGATE launch type.

See `docs/ECS_SERVICE_CONFIG.md` for complete configuration details.

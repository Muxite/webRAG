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
- **`efs-tools.py`**: EFS verification and diagnostic tools (merged verify-efs.py and diagnose-efs.py)
- **`verify-config.py`**: Verify task definition and security group configuration (merged check-task-definition.py and verify-security-groups.py)

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
python ../scripts/build-task-definition.py
```

Generates task definition JSON from `keys.env` and `.env` files.

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
- Resources: 1 vCPU, 2GB RAM
- Desired count: 1
- Capacity provider: FARGATE
- ALB integration: Routes to `gateway` container on port 8080
- Health check grace period: 100 seconds
- Chroma is optional - agent continues working even if Chroma health checks fail

See `docs/ECS_SERVICE_CONFIG.md` for complete configuration details.

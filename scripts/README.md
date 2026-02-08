# Scripts

Short guide to deployment, diagnostics, and audits. Run scripts from `services/` unless noted.

## Common Entry Points
- `deploy.py`: Full deployment (build, push, task defs, services)
- `deploy-autoscale.py`: Deploy autoscale mode (gateway + agent services)
- `deploy-single.py`: Deploy single-service mode
- `check.py`: Health checks for services
- `check-autoscale.py`: Autoscale health checks with extra context

## Configuration and Secrets
- `register-secrets.py`: Sync secrets to AWS Secrets Manager
- `env-to-ecs.py`: Build Secrets Manager payload from env files
- `generate-rabbitmq-cookie.py`: Generate Erlang cookie value

## Network and IAM
- `network_utils.py`: Network validation and fixes
- `network_discovery.py`: Discover VPC/subnet/security group IDs
- `fix-iam-permissions.py`: IAM permissions for ECS task protection
- `fix-vpc-dns.py`: DNS fixes for VPC

## Task Definitions and Infra
- `build-task-definition.py`: Generate ECS task definitions
- `deploy_task_definitions.py`: Register task definitions only
- `deploy_service_discovery.py`: Service discovery setup
- `deploy_network.py`: Network provisioning
- `deploy_ecr.py`: ECR repository setup
- `deploy_common.py`: Shared deploy helpers

## Diagnostics and Audits
- `diagnose-deployment.py`: Deployment diagnostics
- `diagnose-agent-connectivity.py`: Agent connectivity checks
- `diagnose-gateway-failures.py`: Gateway diagnostics
- `audit-aws-changes.py`: Track AWS changes vs git state
- `snapshot-deployment.py`: Capture deployment snapshots
- `comprehensive-audit.py`: Full state audit
- `analyze-audit.py`: Analyze audit snapshots
- `capture-stable-config.py`: Store stable config snapshot
- `verify-config.py`: Validate task definitions and security groups

## EFS Tools
- `efs-tools.py`: EFS verification and diagnostics
- `efs_manager.py`: EFS management utilities

## Requirements
```bash
pip install -r scripts/requirements.txt
```

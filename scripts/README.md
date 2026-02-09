# Scripts

Short guide to deployment, diagnostics, and audits. Run scripts from `services/` unless noted.

## Common Entry Points
- `deploy.py`: Full deployment (build, push, task defs, services)
- `deploy_autoscale.py`: Deploy autoscale mode (gateway + agent services)
- `deploy_single.py`: Deploy single-service mode
- `check.py`: Health checks for services
- `check_autoscale.py`: Autoscale health checks with extra context

## Configuration and Secrets
- `register_secrets.py`: Sync secrets to AWS Secrets Manager
- `update_secrets.py`: Update Secrets Manager from `services/keys.env`
- `env_to_ecs.py`: Build Secrets Manager payload from env files
- `generate_rabbitmq_cookie.py`: Generate Erlang cookie value

## Network and IAM
- `network_utils.py`: Network validation and fixes
- `network_discovery.py`: Discover VPC/subnet/security group IDs
- `fix_iam_permissions.py`: IAM permissions for ECS task protection
- `fix_vpc_dns.py`: DNS fixes for VPC

## Task Definitions and Infra
- `build_task_definition.py`: Generate ECS task definitions
- `deploy_task_definitions.py`: Register task definitions only
- `deploy_service_discovery.py`: Service discovery setup
- `deploy_network.py`: Network provisioning
- `deploy_ecr.py`: ECR repository setup
- `deploy_common.py`: Shared deploy helpers

## Diagnostics and Audits
- `diagnose_deployment.py`: Deployment diagnostics
- `diagnose_agent_connectivity.py`: Agent connectivity checks
- `diagnose_gateway_failures.py`: Gateway diagnostics
- `audit_aws_changes.py`: Track AWS changes vs git state
- `snapshot_deployment.py`: Capture deployment snapshots
- `comprehensive_audit.py`: Full state audit
- `analyze_audit.py`: Analyze audit snapshots
- `capture_stable_config.py`: Store stable config snapshot
- `verify_config.py`: Validate task definitions and security groups

## EFS Tools
- `efs_tools.py`: EFS verification and diagnostics
- `efs_manager.py`: EFS management utilities

## Requirements
```bash
pip install -r scripts/requirements.txt
```


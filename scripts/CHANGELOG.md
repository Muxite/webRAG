# Deployment Scripts Changelog

## 2026-02-01: Service Separation Complete

### Resource Allocations Updated
- **Gateway service**: 0.5 vCPU (512 CPU units), 1GB RAM (1024 MB)
- **Agent service**: 0.25 vCPU (256 CPU units), 0.5GB RAM (512 MB)
- **Single service**: 1 vCPU (1024 CPU units), 2GB RAM (2048 MB) - unchanged

### Scripts Reorganized
- Split `deploy_shared.py` into modular components:
  - `deploy_common.py` - Common utilities (load_aws_config, run_command, get_image_size)
  - `deploy_ecr.py` - ECR operations (push_to_ecr)
  - `deploy_task_definitions.py` - Task definition building
  - `deploy_network.py` - Network configuration
  - `deploy_ecs.py` - ECS service management
  - `deploy_service_discovery.py` - Service discovery setup

### Deployment Mode Enum
- Added `deployment_mode.py` with `DeploymentMode` enum
- All scripts now use enum instead of string comparisons
- Type-safe mode checking across all deployment scripts

### Service Separation
- Separate deployment scripts:
  - `deploy-single.py` - Single service deployment
  - `deploy-autoscale.py` - Autoscale deployment (gateway + agent)
- Service discovery configured for autoscale mode
- Gateway service registered at `euglena-gateway.euglena.local`
- Agent connects to gateway via service discovery DNS

### Status
- Service separation complete and stable (8+ hours)
- Both deployment modes operational
- All services using same Docker images (no `-autoscale` suffix needed)

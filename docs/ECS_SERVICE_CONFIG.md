# ECS Service Configuration Reference

This document contains the exact configuration used for the `euglena-service` ECS service.

## Service Details

- **Cluster**: `euglena-cluster`
- **Launch type / Compute**: AWS Fargate with capacity provider strategy
- **Task definition family**: `euglena`
- **Service name**: `euglena-service`

### Capacity Provider Strategy

- Mode: Use custom (Advanced) strategy
- Capacity provider: `FARGATE`
  - Base: `0`
  - Weight: `1`
- Platform version: `LATEST`

## Deployment Configuration

- **Scheduling strategy**: `REPLICA`
- **Desired tasks (desired count)**: `1`
- **Availability Zone rebalancing**: Enabled
- **Health check grace period**: `100` seconds

## Networking

- **Network mode / ENI type**: `awsvpc` (implied by Fargate)
- **VPC**: `vpc-02cc22c217f55e04b` (`euglena-prod-vpc`)

### Subnets (task placement)

- `subnet-059a0845ba7eb4a09` – `euglena-prod-public-us-east-2a` – AZ: `us-east-2a` – CIDR: `10.0.0.0/20`
- `subnet-0cd121febce7985b9` – `euglena-prod-public-us-east-2b` – AZ: `us-east-2b` – CIDR: `10.0.16.0/20`

### Security Groups

- Mode: Use an existing security group
- Selected security group:
  - ID: `sg-0c45255737f0f9199`
  - Name: `euglena-gateway-sg`

### Public IP

- Auto-assign public IP: Enabled

### Optional Networking Features

- Service Connect: Not configured (disabled)
- Service discovery (Route 53): Not configured (disabled)

## Load Balancing

- **Use load balancing**: Enabled
- **VPC (for load balancer)**: `vpc-02cc22c217f55e04b` (same as service VPC)

### Load Balancer

- Load balancer type: Application Load Balancer (ALB)
- ALB selection mode: Use an existing load balancer
- Selected ALB:
  - Name: `euglena-alb`
  - Scheme: `internet-facing`
  - DNS: `euglena-alb-115605353.us-east-2.elb.amazonaws.com`
  - ARN: `arn:aws:elasticloadbalancing:us-east-2:848960888155:loadbalancer/app/euglena-alb/5f073d0d0520ea72`

### Listener

- Listener selection mode: Use an existing listener
- Listener: `HTTPS:443`
- Listener rule summary:
  - Priority: `default`
  - Rule path: `/`
  - Target group: `euglena-tg`

### Target Group

- Target group selection mode: Use an existing target group
- Target group name: `euglena-tg`
- Health check path: `/health`
- Health check protocol: `HTTP`

### Container Mapping for Load Balancer

- Container: `gateway`
- Host port → container port: `8080:8080`

## Auto Scaling, Volumes, Tags

- Service auto scaling: Not configured (disabled)
- Volumes: None configured at service creation level
- Tags: No additional tags configured

## Notes

- This is a single-service configuration (not multi-service autoscaling setup)
- All containers (chroma, redis, rabbitmq, agent, gateway) run in one task
- Only 1 task should be running at any time
- The service connects to the existing ALB for external access

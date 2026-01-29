# Scripts

## Overview

Scripts directory contains deployment and testing automation:

1. **`deploy.py`**: Complete deployment automation
2. **`check.py`**: Comprehensive health checking
3. **`build-task-definition.py`**: Task definition builder (used by deploy.py)
4. **`network_utils.py`**: Network validation utilities (used by deploy.py)
5. **`fix-iam-permissions.py`**: IAM permission setup (used by deploy.py)
6. **`run_tests.py`**: Test runner for local development (see [TESTING.md](../docs/TESTING.md))

## Quick Start

### Deploy Everything

```bash
cd services
python ../scripts/deploy.py
```

This command performs:
- Docker image build and push to ECR
- ECS task definition generation and registration
- Lambda function packaging (ZIP created in `dist/` directory)
- Network configuration validation and fixes
- Service discovery setup
- Autoscaling rule configuration
- IAM permission updates
- ECS service updates

**Note**: Lambda function code must be deployed manually. The script packages the Lambda function into a ZIP file but does not upload it to AWS. After packaging, manually upload the ZIP from `dist/` directory to AWS Lambda via Console or CLI:
```bash
aws lambda update-function-code --function-name <function-name> --zip-file fileb://dist/lambda-autoscaling-<timestamp>.zip
```

### Check System Health

```bash
cd services
python ../scripts/check.py
```

This checks:
- ECS service status and task counts
- Container health status
- Network configuration and connectivity
- Agent registration and connectivity
- Service discovery DNS resolution

## Deploy Options

```bash
# Full deployment (default)
python ../scripts/deploy.py

# Deploy specific service
python ../scripts/deploy.py --service gateway
python ../scripts/deploy.py --service agent

# Skip steps for faster updates
python ../scripts/deploy.py --skip-ecr          # Skip image push
python ../scripts/deploy.py --skip-lambda       # Skip Lambda packaging
python ../scripts/deploy.py --skip-network-check  # Skip network validation

# Wait for services to stabilize after deployment
python ../scripts/deploy.py --wait
```

## Check Options

```bash
# Full health check (default)
python ../scripts/check.py

# Check specific service
python ../scripts/check.py --service gateway
python ../scripts/check.py --service agent

# Verbose output with detailed information
python ../scripts/check.py --verbose
```

## Configuration

Scripts read configuration from `services/` directory:
- `aws.env`: AWS configuration (account ID, region, cluster name, service names)
- `.env`: Application environment variables
- `keys.env`: Secret keys (loaded from AWS Secrets Manager in production, not committed to repository)

## Architecture

Scripts use functional approach:
- No object-oriented overhead
- Modular design with shared utilities
- Integrated functionality consolidated into main scripts
- Self-contained scripts handle their own dependencies

## Lambda Deployment

The deploy script packages the Lambda function but does not automatically deploy it. After running `deploy.py`, manually upload the Lambda package:

1. Find the latest ZIP file in `dist/` directory (e.g., `lambda-autoscaling-YYYYMMDDHHMMSS.zip`)
2. Upload via AWS Console or CLI:
   ```bash
   aws lambda update-function-code \
     --function-name euglena-autoscaling \
     --zip-file fileb://dist/lambda-autoscaling-YYYYMMDDHHMMSS.zip \
     --region <region>
   ```

## Troubleshooting

### View Logs

```bash
aws logs tail /ecs/euglena --follow
```

### Re-deploy After Issues

```bash
python ../scripts/deploy.py --wait
```

### Check Status

```bash
python ../scripts/check.py --verbose
```

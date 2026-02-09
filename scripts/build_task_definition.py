"""
Builds and registers ECS task definitions.
Supports single service or autoscale gateway/agent deployment modes.
"""
import json
import copy
import os
import subprocess
import sys
import argparse
from pathlib import Path
from dotenv import dotenv_values


def load_aws_config(base_dir):
    """
    Load AWS configuration from aws.env.
    :param base_dir: Base directory to search for aws.env.
    :returns: Dictionary of AWS environment variables.
    """
    aws_env_path = base_dir / "aws.env"
    aws_env = {}
    
    if aws_env_path.exists():
        aws_env = dict(dotenv_values(str(aws_env_path)))
    
    return aws_env


def load_secrets_from_keys_env(services_dir):
    """
    Load secret keys from keys.env.
    :param services_dir: Services directory path where keys.env should be located.
    :returns: List of secret key names.
    """
    keys_env_path = services_dir / "keys.env"
    secret_keys = []
    
    if keys_env_path.exists():
        keys_env = dict(dotenv_values(str(keys_env_path)))
        secret_keys = [key for key in keys_env.keys() if key and not key.startswith("#")]
    
    return secret_keys


def load_env_variables(services_dir, secret_keys):
    """
    Load environment variables from .env, excluding secrets.
    :param services_dir: Services directory path where .env should be located.
    :param secret_keys: List of secret key names to exclude.
    :returns: Dictionary of environment variables.
    """
    env_path = services_dir / ".env"
    env_vars = {}
    
    if env_path.exists():
        env_dict = dict(dotenv_values(str(env_path)))
        
        for key, value in env_dict.items():
            if key and value and not key.startswith("#"):
                if key not in secret_keys:
                    env_vars[key] = value
    
    return env_vars


def apply_version_overrides(env_vars):
    """
    Apply version-related environment overrides.
    :param env_vars: Environment variables dictionary.
    :returns: Updated environment variables dictionary.
    """
    deployment_number = os.environ.get("DEPLOYMENT_NUMBER")
    variant_number = os.environ.get("VARIANT_NUMBER")
    if variant_number:
        env_vars["VARIANT_NUMBER"] = variant_number
    if deployment_number:
        env_vars["DEPLOYMENT_NUMBER"] = deployment_number
    return env_vars


def build_environment_list(env_vars):
    """
    Convert env vars to ECS environment list format.
    :param env_vars: Dictionary of environment variables.
    :returns: List of {"name": key, "value": value} dictionaries.
    """
    env_ecs = {}
    for k, v in env_vars.items():
        value = str(v)
        if k in ("RABBITMQ_URL", "REDIS_URL", "CHROMA_URL", "GATEWAY_URL"):
            value = value.replace("@rabbitmq:", "@localhost:").replace("@redis:", "@localhost:").replace("@chroma:", "@localhost:").replace("@gateway:", "@localhost:")
            value = value.replace("://rabbitmq:", "://localhost:").replace("://redis:", "://localhost:").replace("://chroma:", "://localhost:").replace("://gateway:", "://localhost:")
        env_ecs[k] = value
    
    return [{"name": k, "value": v} for k, v in sorted(env_ecs.items())]


def redact_task_definition(task_def):
    """
    Redact sensitive fields for sharing.
    :param task_def: Task definition dictionary.
    :returns: Redacted task definition dictionary.
    """
    redacted = copy.deepcopy(task_def)
    
    if "taskRoleArn" in redacted:
        redacted["taskRoleArn"] = "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole"
    if "executionRoleArn" in redacted:
        redacted["executionRoleArn"] = "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole"
    
    for container in redacted.get("containerDefinitions", []):
        if "image" in container:
            image = container["image"]
            if ".dkr.ecr." in image:
                parts = image.split("/")
                if len(parts) >= 2:
                    service_name = parts[-1]
                    container["image"] = f"ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/euglena/{service_name}"
                else:
                    container["image"] = "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/euglena/SERVICE"
        
        if "logConfiguration" in container and "options" in container["logConfiguration"]:
            log_opts = container["logConfiguration"]["options"]
            if "awslogs-region" in log_opts:
                log_opts["awslogs-region"] = "REGION"
        
        if "secrets" in container:
            for secret in container["secrets"]:
                if "valueFrom" in secret:
                    key_name = secret.get("name", "KEY")
                    secret["valueFrom"] = f"arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:SECRET_NAME-SUFFIX:{key_name}::"
    
    return redacted


def extract_arn_suffix(secret_input):
    """
    Extract ARN suffix or pass through if already a suffix.
    :param secret_input: Secret ARN or suffix string.
    :returns: Extracted suffix or original input.
    """
    if not secret_input.startswith("arn:aws:secretsmanager:"):
        return secret_input
    
    secret_idx = secret_input.find("secret:")
    if secret_idx == -1:
        return secret_input
    
    after_secret = secret_input[secret_idx + 7:]
    name_suffix_part = after_secret.split(":")[0] if ":" in after_secret else after_secret
    
    if "-" in name_suffix_part:
        return name_suffix_part[name_suffix_part.rfind("-") + 1:]
    
    return secret_input


def build_secrets_list(secret_name, secret_arn_suffix, region, account_id, secret_keys):
    """
    Build ECS secrets entries from keys.env names.
    :param secret_name: Secrets Manager secret name.
    :param secret_arn_suffix: ARN suffix for the secret.
    :param region: AWS region.
    :param account_id: AWS account ID.
    :param secret_keys: List of secret key names to include.
    :returns: List of secret dicts for ECS task definition.
    """
    secrets = []
    if secret_arn_suffix and secret_keys:
        for key in sorted(secret_keys):
            secrets.append({
                "name": key,
                "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-{secret_arn_suffix}:{key}::"
            })
    return secrets


def _make_health_check(command, start_period=300, interval=60, retries=5, timeout=10):
    """
    Create health check config for ECS.
    :param command: Health check shell command.
    :param start_period: Grace period before health checks start (seconds).
    :param interval: Time between health checks (seconds).
    :param retries: Consecutive failures before marking unhealthy.
    :param timeout: Maximum time for health check to complete (seconds).
    :returns: Health check configuration dictionary.
    """
    return {
        "command": ["CMD-SHELL", command],
        "interval": interval,
        "retries": retries,
        "startPeriod": start_period,
        "timeout": timeout
    }


def _make_log_config(prefix, region):
    """
    Create CloudWatch log config for container logging.
    :param prefix: Log stream prefix.
    :param region: AWS region.
    :returns: Log configuration dictionary.
    """
    return {
        "logDriver": "awslogs",
        "options": {
            "awslogs-group": "/ecs/euglena",
            "awslogs-region": region,
            "awslogs-stream-prefix": prefix
        }
    }


def build_gateway_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env):
    """
    Build gateway task definition for gateway + deps.
    :param account_id: AWS account ID.
    :param region: AWS region.
    :param secret_name: Secrets Manager secret name.
    :param secret_arn_suffix: ARN suffix for the secret.
    :param secret_keys: List of secret key names.
    :param env_vars: Dictionary of environment variables from .env.
    :param aws_env: Dictionary of AWS environment variables from aws.env.
    :returns: Task definition dictionary.
    """
    secrets = build_secrets_list(secret_name, secret_arn_suffix, region, account_id, secret_keys)
    
    rabbitmq_secrets = []
    if secret_arn_suffix and "RABBITMQ_ERLANG_COOKIE" in secret_keys:
        rabbitmq_secrets.append({
            "name": "RABBITMQ_ERLANG_COOKIE",
            "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-{secret_arn_suffix}:RABBITMQ_ERLANG_COOKIE::"
        })
    
    shared_efs_id = aws_env.get("EFS_FILE_SYSTEM_ID", "").strip() if aws_env else ""
    chroma_efs_id = aws_env.get("CHROMA_EFS_FILE_SYSTEM_ID", shared_efs_id).strip() if aws_env else shared_efs_id
    rabbitmq_efs_id = aws_env.get("RABBITMQ_EFS_FILE_SYSTEM_ID", shared_efs_id).strip() if aws_env else shared_efs_id
    
    chroma_efs_id = chroma_efs_id if chroma_efs_id else None
    rabbitmq_efs_id = rabbitmq_efs_id if rabbitmq_efs_id else None
    
    chroma_health_command = "curl -f -s -S --max-time 15 --connect-timeout 5 http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1 || exit 1"
    chroma_mount_points = []
    if chroma_efs_id:
        chroma_mount_points.append({"sourceVolume": "chroma-data", "containerPath": "/chroma-data", "readOnly": False})
    
    chroma_env = [
        {"name": "IS_PERSISTENT", "value": "TRUE"},
        {"name": "PERSIST_DIRECTORY", "value": "/chroma-data"}
    ]
    if chroma_efs_id:
        chroma_env.extend([
            {"name": "SENTENCE_TRANSFORMERS_HOME", "value": "/chroma-data/.cache"},
            {"name": "TRANSFORMERS_CACHE", "value": "/chroma-data/.cache/huggingface"}
        ])
    
    chroma = {
        "cpu": 0,
        "environment": chroma_env,
        "essential": False,
        "healthCheck": _make_health_check(chroma_health_command, start_period=300, interval=60, retries=6, timeout=30),
        "image": "chromadb/chroma:latest",
        "logConfiguration": _make_log_config("chroma", region),
        "mountPoints": chroma_mount_points,
        "name": "chroma",
        "portMappings": [{"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}],
        "systemControls": [],
        "volumesFrom": []
    }
    
    redis = {
        "command": ["redis-server"],
        "cpu": 0,
        "environment": [],
        "essential": True,
        "healthCheck": _make_health_check("redis-cli ping | grep PONG || exit 1", start_period=120, interval=90, retries=6, timeout=15),
        "image": "redis:7-alpine",
        "logConfiguration": _make_log_config("redis", region),
        "mountPoints": [],
        "name": "redis",
        "portMappings": [{"containerPort": 6379, "hostPort": 6379, "protocol": "tcp"}],
        "systemControls": [],
        "volumesFrom": []
    }
    
    rabbitmq_mount_points = []
    if rabbitmq_efs_id:
        rabbitmq_mount_points.append({"sourceVolume": "rabbitmq-data", "containerPath": "/var/lib/rabbitmq", "readOnly": False})
    
    rabbitmq = {
        "command": [
            "sh",
            "-c",
            "if [ -n \"$RABBITMQ_ERLANG_COOKIE\" ]; then mkdir -p /var/lib/rabbitmq && echo \"$RABBITMQ_ERLANG_COOKIE\" > /var/lib/rabbitmq/.erlang.cookie && chmod 600 /var/lib/rabbitmq/.erlang.cookie && chown rabbitmq:rabbitmq /var/lib/rabbitmq/.erlang.cookie; fi && exec /usr/local/bin/docker-entrypoint.sh rabbitmq-server"
        ],
        "cpu": 0,
        "environment": [],
        "essential": True,
        "healthCheck": _make_health_check("rabbitmq-diagnostics ping || exit 1", start_period=300, interval=90, retries=6, timeout=15),
        "image": "rabbitmq:3-management",
        "logConfiguration": _make_log_config("rabbitmq", region),
        "mountPoints": rabbitmq_mount_points,
        "name": "rabbitmq",
        "portMappings": [
            {"containerPort": 5672, "hostPort": 5672, "protocol": "tcp"},
            {"containerPort": 15672, "hostPort": 15672, "protocol": "tcp"}
        ],
        "secrets": rabbitmq_secrets,
        "systemControls": [],
        "volumesFrom": []
    }
    
    env_list = build_environment_list(env_vars)
    env_list.append({"name": "GATEWAY_DEBUG_QUEUE_NAME", "value": env_vars.get("GATEWAY_DEBUG_QUEUE_NAME", "gateway.debug")})
    env_list.append({"name": "GATEWAY_DEBUG_QUEUE_PHRASE", "value": env_vars.get("GATEWAY_DEBUG_QUEUE_PHRASE", "debugdebugdebug")})
    
    gateway = {
        "cpu": 0,
        "dependsOn": [
            {"condition": "START", "containerName": "redis"},
            {"condition": "START", "containerName": "rabbitmq"}
        ],
        "environment": env_list,
        "essential": True,
        "healthCheck": _make_health_check("curl -f http://localhost:8080/health || exit 1", start_period=300, interval=90, retries=6, timeout=15),
        "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/gateway:latest",
        "logConfiguration": _make_log_config("gateway", region),
        "mountPoints": [],
        "name": "gateway",
        "portMappings": [{"containerPort": 8080, "hostPort": 8080, "protocol": "tcp"}],
        "secrets": secrets,
        "systemControls": [],
        "volumesFrom": []
    }
    
    rabbitmq_url_original = env_vars.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    
    if "@" in rabbitmq_url_original:
        creds_and_rest = rabbitmq_url_original.split("@", 1)
        creds = creds_and_rest[0]
        rest = creds_and_rest[1]
        if ":" in rest:
            port_and_path = rest.split(":", 1)[1]
            rabbitmq_url = f"{creds}@localhost:{port_and_path}"
        else:
            rabbitmq_url = f"{creds}@localhost:5672/"
    else:
        scheme = rabbitmq_url_original.split("://")[0] if "://" in rabbitmq_url_original else "amqp"
        rabbitmq_url = f"{scheme}://localhost:5672/"
    
    metrics_env = [
        {"name": "RABBITMQ_URL", "value": rabbitmq_url},
        {"name": "PUBLISH_QUEUE_DEPTH_METRICS", "value": "true"},
        {"name": "QUEUE_DEPTH_METRICS_INTERVAL", "value": "1"},
        {"name": "CLOUDWATCH_NAMESPACE", "value": "Euglena/RabbitMQ"},
        {"name": "QUEUE_NAME", "value": env_vars.get("AGENT_INPUT_QUEUE", "agent.mandates")},
        {"name": "GATEWAY_DEBUG_QUEUE_NAME", "value": env_vars.get("GATEWAY_DEBUG_QUEUE_NAME", "gateway.debug")}
    ]
    
    metrics = {
        "cpu": 0,
        "dependsOn": [
            {"condition": "START", "containerName": "rabbitmq"}
        ],
        "environment": metrics_env,
        "essential": False,
        "healthCheck": _make_health_check("curl -f http://localhost:8082/health || exit 1", start_period=60, interval=30, retries=3, timeout=10),
        "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/metrics:latest",
        "logConfiguration": _make_log_config("metrics", region),
        "mountPoints": [],
        "name": "metrics",
        "portMappings": [{"containerPort": 8082, "hostPort": 8082, "protocol": "tcp"}],
        "systemControls": [],
        "volumesFrom": []
    }
    
    containers = [chroma, redis, rabbitmq, gateway, metrics]
    
    volumes = []
    using_shared_efs = (chroma_efs_id and rabbitmq_efs_id and 
                        chroma_efs_id == rabbitmq_efs_id)
    
    if chroma_efs_id:
        efs_config = {
            "fileSystemId": chroma_efs_id,
            "transitEncryption": "ENABLED",
            "authorizationConfig": {
                "iam": "ENABLED"
            }
        }
        volumes.append({
            "name": "chroma-data",
            "efsVolumeConfiguration": efs_config
        })
    
    if rabbitmq_efs_id:
        efs_config = {
            "fileSystemId": rabbitmq_efs_id,
            "transitEncryption": "ENABLED",
            "authorizationConfig": {
                "iam": "ENABLED"
            }
        }
        volumes.append({
            "name": "rabbitmq-data",
            "efsVolumeConfiguration": efs_config
        })
    
    return {
        "family": "euglena-gateway",
        "containerDefinitions": containers,
        "taskRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskRole",
        "executionRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole",
        "networkMode": "awsvpc",
        "volumes": volumes,
        "placementConstraints": [],
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "1024",
        "memory": "2048"
    }


def build_agent_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env, gateway_host="euglena-gateway.euglena.local"):
    """
    Build agent task definition for ECS.
    :param account_id: AWS account ID.
    :param region: AWS region.
    :param secret_name: Secrets Manager secret name.
    :param secret_arn_suffix: ARN suffix for the secret.
    :param secret_keys: List of secret key names.
    :param env_vars: Dictionary of environment variables from .env.
    :param aws_env: Dictionary of AWS environment variables from aws.env.
    :param gateway_host: Hostname for gateway service.
    :returns: Task definition dictionary.
    """
    secrets = build_secrets_list(secret_name, secret_arn_suffix, region, account_id, secret_keys)
    
    agent_env_vars = env_vars.copy()
    
    env_list = build_environment_list(agent_env_vars)
    
    for env_item in env_list:
        key = env_item.get("name", "")
        value = env_item.get("value", "")
        if key in ("RABBITMQ_URL", "REDIS_URL", "CHROMA_URL", "GATEWAY_URL"):
            value = value.replace("@localhost:", f"@{gateway_host}:")
            value = value.replace("://localhost:", f"://{gateway_host}:")
            value = value.replace("localhost:", f"{gateway_host}:")
            value = value.replace("http://localhost", f"http://{gateway_host}")
            value = value.replace("https://localhost", f"https://{gateway_host}")
            value = value.replace("amqp://localhost", f"amqp://{gateway_host}")
            value = value.replace("redis://localhost", f"redis://{gateway_host}")
            env_item["value"] = value
    
    agent = {
        "cpu": 0,
        "environment": env_list,
        "essential": True,
        "healthCheck": _make_health_check("curl -f http://localhost:8081/health || exit 1", start_period=300, interval=90, retries=6, timeout=15),
        "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/agent:latest",
        "logConfiguration": _make_log_config("agent", region),
        "mountPoints": [],
        "name": "agent",
        "portMappings": [{"containerPort": 8081, "hostPort": 8081, "protocol": "tcp"}],
        "secrets": secrets,
        "systemControls": [],
        "volumesFrom": []
    }
    
    return {
        "family": "euglena-agent",
        "containerDefinitions": [agent],
        "taskRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskRole",
        "executionRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole",
        "networkMode": "awsvpc",
        "volumes": [],
        "placementConstraints": [],
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "256",
        "memory": "512"
    }


def build_euglena_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env):
    """
    Build single-service task definition with all containers.
    :param account_id: AWS account ID.
    :param region: AWS region.
    :param secret_name: Secrets Manager secret name.
    :param secret_arn_suffix: ARN suffix for the secret.
    :param secret_keys: List of secret key names.
    :param env_vars: Dictionary of environment variables from .env.
    :param aws_env: Dictionary of AWS environment variables from aws.env.
    :returns: Task definition dictionary.
    """
    secrets = build_secrets_list(secret_name, secret_arn_suffix, region, account_id, secret_keys)
    
    rabbitmq_secrets = []
    if secret_arn_suffix and "RABBITMQ_ERLANG_COOKIE" in secret_keys:
        rabbitmq_secrets.append({
            "name": "RABBITMQ_ERLANG_COOKIE",
            "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-{secret_arn_suffix}:RABBITMQ_ERLANG_COOKIE::"
        })
    
    shared_efs_id = aws_env.get("EFS_FILE_SYSTEM_ID", "").strip() if aws_env else ""
    chroma_efs_id = aws_env.get("CHROMA_EFS_FILE_SYSTEM_ID", shared_efs_id).strip() if aws_env else shared_efs_id
    rabbitmq_efs_id = aws_env.get("RABBITMQ_EFS_FILE_SYSTEM_ID", shared_efs_id).strip() if aws_env else shared_efs_id
    
    chroma_efs_id = chroma_efs_id if chroma_efs_id else None
    rabbitmq_efs_id = rabbitmq_efs_id if rabbitmq_efs_id else None
    
    chroma_health_command = "curl -f -s -S --max-time 15 --connect-timeout 5 http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1 || exit 1"
    chroma_mount_points = []
    if chroma_efs_id:
        chroma_mount_points.append({"sourceVolume": "chroma-data", "containerPath": "/chroma-data", "readOnly": False})
    
    chroma_env = [
        {"name": "IS_PERSISTENT", "value": "TRUE"},
        {"name": "PERSIST_DIRECTORY", "value": "/chroma-data"}
    ]
    if chroma_efs_id:
        chroma_env.extend([
            {"name": "SENTENCE_TRANSFORMERS_HOME", "value": "/chroma-data/.cache"},
            {"name": "TRANSFORMERS_CACHE", "value": "/chroma-data/.cache/huggingface"}
        ])
    
    chroma = {
        "cpu": 0,
        "environment": chroma_env,
        "essential": False,
        "healthCheck": _make_health_check(chroma_health_command, start_period=300, interval=60, retries=6, timeout=30),
        "image": "chromadb/chroma:latest",
        "logConfiguration": _make_log_config("chroma", region),
        "mountPoints": chroma_mount_points,
        "name": "chroma",
        "portMappings": [{"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}],
        "systemControls": [],
        "volumesFrom": []
    }
    
    redis = {
        "command": ["redis-server"],
        "cpu": 0,
        "environment": [],
        "essential": True,
        "healthCheck": _make_health_check("redis-cli ping | grep PONG || exit 1", start_period=120, interval=90, retries=6, timeout=15),
        "image": "redis:7-alpine",
        "logConfiguration": _make_log_config("redis", region),
        "mountPoints": [],
        "name": "redis",
        "portMappings": [{"containerPort": 6379, "hostPort": 6379, "protocol": "tcp"}],
        "systemControls": [],
        "volumesFrom": []
    }
    
    rabbitmq_mount_points = []
    if rabbitmq_efs_id:
        rabbitmq_mount_points.append({"sourceVolume": "rabbitmq-data", "containerPath": "/var/lib/rabbitmq", "readOnly": False})
    
    rabbitmq = {
        "command": [
            "sh",
            "-c",
            "if [ -n \"$RABBITMQ_ERLANG_COOKIE\" ]; then mkdir -p /var/lib/rabbitmq && echo \"$RABBITMQ_ERLANG_COOKIE\" > /var/lib/rabbitmq/.erlang.cookie && chmod 600 /var/lib/rabbitmq/.erlang.cookie && chown rabbitmq:rabbitmq /var/lib/rabbitmq/.erlang.cookie; fi && exec /usr/local/bin/docker-entrypoint.sh rabbitmq-server"
        ],
        "cpu": 0,
        "environment": [],
        "essential": True,
        "healthCheck": _make_health_check("rabbitmq-diagnostics ping || exit 1", start_period=300, interval=90, retries=6, timeout=15),
        "image": "rabbitmq:3-management",
        "logConfiguration": _make_log_config("rabbitmq", region),
        "mountPoints": rabbitmq_mount_points,
        "name": "rabbitmq",
        "portMappings": [
            {"containerPort": 5672, "hostPort": 5672, "protocol": "tcp"},
            {"containerPort": 15672, "hostPort": 15672, "protocol": "tcp"}
        ],
        "secrets": rabbitmq_secrets,
        "systemControls": [],
        "volumesFrom": []
    }
    
    env_list = build_environment_list(env_vars)
    
    agent = {
        "cpu": 0,
        "dependsOn": [
            {"condition": "START", "containerName": "redis"},
            {"condition": "START", "containerName": "rabbitmq"}
        ],
        "environment": env_list,
        "essential": True,
        "healthCheck": _make_health_check("curl -f http://localhost:8081/health || exit 1", start_period=300, interval=90, retries=6, timeout=15),
        "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/agent:latest",
        "logConfiguration": _make_log_config("agent", region),
        "mountPoints": [],
        "name": "agent",
        "portMappings": [{"containerPort": 8081, "hostPort": 8081, "protocol": "tcp"}],
        "secrets": secrets,
        "systemControls": [],
        "volumesFrom": []
    }
    
    gateway = {
        "cpu": 0,
        "dependsOn": [
            {"condition": "START", "containerName": "chroma"},
            {"condition": "START", "containerName": "redis"},
            {"condition": "START", "containerName": "rabbitmq"},
            {"condition": "START", "containerName": "agent"}
        ],
        "environment": env_list,
        "essential": True,
        "healthCheck": _make_health_check("curl -f http://localhost:8080/health || exit 1", start_period=300, interval=90, retries=6, timeout=15),
        "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/gateway:latest",
        "logConfiguration": _make_log_config("gateway", region),
        "mountPoints": [],
        "name": "gateway",
        "portMappings": [{"containerPort": 8080, "hostPort": 8080, "protocol": "tcp"}],
        "secrets": secrets,
        "systemControls": [],
        "volumesFrom": []
    }
    
    chroma_efs_id = chroma_efs_id if chroma_efs_id else None
    rabbitmq_efs_id = rabbitmq_efs_id if rabbitmq_efs_id else None
    
    containers = [chroma, redis, rabbitmq, agent, gateway]
    
    volumes = []
    using_shared_efs = (chroma_efs_id and rabbitmq_efs_id and 
                        chroma_efs_id == rabbitmq_efs_id)
    
    if chroma_efs_id:
        efs_config = {
            "fileSystemId": chroma_efs_id,
            "transitEncryption": "ENABLED",
            "authorizationConfig": {
                "iam": "ENABLED"
            }
        }
        volumes.append({
            "name": "chroma-data",
            "efsVolumeConfiguration": efs_config
        })
    
    if rabbitmq_efs_id:
        efs_config = {
            "fileSystemId": rabbitmq_efs_id,
            "transitEncryption": "ENABLED",
            "authorizationConfig": {
                "iam": "ENABLED"
            }
        }
        volumes.append({
            "name": "rabbitmq-data",
            "efsVolumeConfiguration": efs_config
        })
    
    return {
        "family": "euglena",
        "containerDefinitions": containers,
        "taskRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskRole",
        "executionRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole",
        "networkMode": "awsvpc",
        "volumes": volumes,
        "placementConstraints": [],
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "512",
        "memory": "1024"
    }


def _get_aws_config(aws_env):
    """
    Get AWS config values from aws.env or prompt.
    :param aws_env: AWS environment variables.
    :returns: Tuple of (account_id, region, secret_name, secret_arn_suffix).
    """
    def get_or_prompt(key, prompt_text):
        value = aws_env.get(key, "").strip() if aws_env else ""
        return value if value else input(f"{prompt_text}: ").strip()
    
    account_id = get_or_prompt("AWS_ACCOUNT_ID", "Account ID")
    region = get_or_prompt("AWS_REGION", "Region")
    secret_name = get_or_prompt("AWS_SECRET_NAME", "Secret name")
    secret_arn_suffix = get_or_prompt("AWS_SECRET_ARN_SUFFIX", "Secret ARN suffix") or ""
    
    return account_id, region, secret_name, secret_arn_suffix


def _write_task_definition(base_dir, task_def, name):
    """
    Write task definition JSON files.
    :param base_dir: Base directory to write files.
    :param task_def: Task definition dictionary.
    :param name: Task definition name.
    :returns: Tuple of (path, redacted_path).
    """
    path = base_dir / f"task-definition-{name}.json"
    if path.exists():
        path.unlink()
    with open(path, 'w') as f:
        json.dump(task_def, f, indent=4)
    
    redacted_path = base_dir / f"task-definition-{name}.redacted.json"
    if redacted_path.exists():
        redacted_path.unlink()
    with open(redacted_path, 'w') as f:
        json.dump(redact_task_definition(task_def), f, indent=4)
    
    return path, redacted_path


def run_command(cmd, check=True):
    """
    Run a shell command and return success status.
    :param cmd: Command as list of strings.
    :param check: Whether to raise on non-zero exit.
    :returns: True on success, False on failure.
    """
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False


def register_task_definition(task_def_path, region):
    """
    Register task definition with AWS ECS.
    :param task_def_path: Path to task definition JSON file.
    :param region: AWS region.
    :returns: True on success, False on failure.
    """
    abs_path = task_def_path.resolve()
    file_url = f"file://{abs_path.as_posix()}"
    
    cmd = [
        "aws", "ecs", "register-task-definition",
        "--cli-input-json", file_url,
        "--region", region
    ]
    
    if run_command(cmd):
        print(f"Registered: {task_def_path.name}")
        return True
    print(f"Register failed: {task_def_path.name}")
    return False


def parse_args():
    """
    Parse CLI arguments.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build and register ECS task definitions")
    try:
        from scripts.deployment_mode import DeploymentMode
    except ImportError:
        from deployment_mode import DeploymentMode
    
    parser.add_argument("--mode", choices=["single", "autoscale"], default="single",
                       help="Deployment mode: single (all containers) or autoscale (gateway/agent)")
    
    return parser.parse_args()

def main():
    """
    Generate and register ECS task definitions.
    :returns: None.
    """
    args = parse_args()
    try:
        from scripts.deployment_mode import DeploymentMode
    except ImportError:
        from deployment_mode import DeploymentMode
    mode = DeploymentMode.from_string(args.mode)
    
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "services" / "task-definitions"
    base_dir.mkdir(parents=True, exist_ok=True)
    services_dir = repo_root / "services"
    
    aws_env = load_aws_config(services_dir)
    account_id, region, secret_name, secret_arn_suffix = _get_aws_config(aws_env)
    
    if not account_id or not region:
        print("Account ID and Region required.")
        sys.exit(1)
    
    secret_keys = load_secrets_from_keys_env(services_dir)
    env_vars = load_env_variables(services_dir, secret_keys)
    env_vars = apply_version_overrides(env_vars)
    
    if not secret_arn_suffix or not secret_arn_suffix.strip():
        if aws_env.get("AWS_SECRET_ARN_SUFFIX"):
            secret_arn_suffix = aws_env.get("AWS_SECRET_ARN_SUFFIX").strip()
        elif secret_keys:
            secret_input = input("Secret ARN or suffix: ").strip()
            if secret_input:
                secret_arn_suffix = extract_arn_suffix(secret_input)
    
    if mode == DeploymentMode.SINGLE:
        print("Generate single service task definition")
        task_def = build_euglena_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env)
        task_path, task_redacted_path = _write_task_definition(base_dir, task_def, "euglena")
        print(f"Generated: {task_path}")
        
        print("\nRegister task definition")
        success = register_task_definition(task_path, region)
        
        if not success:
            print("\nRegister failed")
            sys.exit(1)
        print("\nRegister ok")
    else:
        print("Generate gateway task definition")
        gateway_task_def = build_gateway_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env)
        gateway_path, gateway_redacted_path = _write_task_definition(base_dir, gateway_task_def, "euglena-gateway")
        print(f"Generated: {gateway_path}")
        
        print("\nRegister gateway task definition")
        gateway_success = register_task_definition(gateway_path, region)
        
        print("Generate agent task definition")
        agent_task_def = build_agent_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env)
        agent_path, agent_redacted_path = _write_task_definition(base_dir, agent_task_def, "euglena-agent")
        print(f"Generated: {agent_path}")
        
        print("\nRegister agent task definition")
        agent_success = register_task_definition(agent_path, region)
        
        if not gateway_success or not agent_success:
            print("\nRegister failed")
            sys.exit(1)
        print("\nRegister ok")
    
    from datetime import datetime
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

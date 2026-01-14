import json
import copy
from pathlib import Path
from dotenv import dotenv_values


def load_aws_config(base_dir):
    """
    Loads AWS configuration from aws.env file.
    :param base_dir: Base directory to search for aws.env
    :returns: Dictionary of AWS environment variables
    """
    aws_env_path = base_dir / "aws.env"
    aws_env = {}
    
    if aws_env_path.exists():
        aws_env = dict(dotenv_values(str(aws_env_path)))
    
    return aws_env


def load_secrets_from_keys_env(services_dir):
    """
    Loads secret keys from keys.env file.
    :param services_dir: Services directory path where keys.env should be located
    :returns: List of secret key names
    """
    keys_env_path = services_dir / "keys.env"
    secret_keys = []
    
    if keys_env_path.exists():
        keys_env = dict(dotenv_values(str(keys_env_path)))
        secret_keys = [key for key in keys_env.keys() if key and keys_env[key] and not key.startswith("#")]
    
    return secret_keys


def load_env_variables(services_dir, secret_keys):
    """
    Loads environment variables from .env file, excluding secrets and Lambda variables.
    :param services_dir: Services directory path where .env should be located
    :param secret_keys: List of secret key names to exclude
    :returns: Dictionary of environment variables
    """
    env_path = services_dir / ".env"
    env_vars = {}
    
    if env_path.exists():
        env_dict = dict(dotenv_values(str(env_path)))
        lambda_vars = {"MIN_WORKERS", "MAX_WORKERS", "TARGET_MESSAGES_PER_WORKER"}
        
        for key, value in env_dict.items():
            if key and value and not key.startswith("#"):
                if key not in secret_keys and key not in lambda_vars:
                    env_vars[key] = value
    
    return env_vars


def build_environment_list(env_vars):
    """
    Converts environment variables dictionary to ECS environment list format.
    :param env_vars: Dictionary of environment variables
    :returns: List of {"name": key, "value": value} dictionaries
    """
    return [{"name": k, "value": v} for k, v in sorted(env_vars.items())]


def redact_task_definition(task_def):
    """Redact task definition with placeholders for account/region."""
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
    Extracts ARN suffix from Secrets Manager ARN or returns as-is.
    :param secret_input: Secret ARN or suffix string
    :returns: Extracted suffix or original input
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
    Builds secrets list from secret keys for ECS task definition.
    :param secret_name: Secrets Manager secret name
    :param secret_arn_suffix: ARN suffix for the secret
    :param region: AWS region
    :param account_id: AWS account ID
    :param secret_keys: List of secret key names to include
    :returns: List of secret dicts for ECS task definition
    """
    secrets = []
    if secret_arn_suffix and secret_keys:
        for key in sorted(secret_keys):
            secrets.append({
                "name": key,
                "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-{secret_arn_suffix}:{key}::"
            })
    return secrets


def _make_health_check(command, start_period=300, interval=30, retries=3, timeout=5):
    """
    Creates health check config using command with specified timing.

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
    """Creates CloudWatch log config for container logging."""
    return {
        "logDriver": "awslogs",
        "options": {
            "awslogs-group": "/ecs/euglena",
            "awslogs-region": region,
            "awslogs-stream-prefix": prefix
        }
    }


def _make_base_container(name, image, port, health_command, region, cpu=0, essential=True, start_period=300):
    """Creates base container definition with health check and logging."""
    return {
        "cpu": cpu,
        "essential": essential,
        "healthCheck": _make_health_check(health_command, start_period=start_period),
        "image": image,
        "logConfiguration": _make_log_config(name, region),
        "mountPoints": [],
        "name": name,
        "portMappings": [{"containerPort": port, "protocol": "tcp"}] if port else [],
        "systemControls": [],
        "volumesFrom": []
    }


def build_gateway_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars):
    """
    Builds gateway task definition with redis, rabbitmq, chroma sidecars.
    :param account_id: AWS account ID
    :param region: AWS region
    :param secret_name: Secrets Manager secret name
    :param secret_arn_suffix: ARN suffix for the secret
    :param secret_keys: List of secret key names
    :param env_vars: Dictionary of environment variables from .env
    :returns: Task definition dictionary
    """
    gateway_secrets = build_secrets_list(secret_name, secret_arn_suffix, region, account_id, secret_keys)
    
    rabbitmq_secrets = []
    if secret_arn_suffix and "RABBITMQ_ERLANG_COOKIE" in secret_keys:
        rabbitmq_secrets.append({
            "name": "RABBITMQ_ERLANG_COOKIE",
            "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-{secret_arn_suffix}:RABBITMQ_ERLANG_COOKIE::"
        })
    
    chroma = _make_base_container("chroma", "chromadb/chroma:latest", 8000, "curl -f http://localhost:8000/api/v1/heartbeat || exit 1", region, start_period=240)
    chroma["environment"] = [{"name": "IS_PERSISTENT", "value": "TRUE"}, {"name": "PERSIST_DIRECTORY", "value": "/chroma-data"}]
    
    redis = _make_base_container("redis", "redis:7-alpine", 6379, "redis-cli ping | grep PONG || exit 1", region, start_period=120)
    redis["command"] = ["redis-server"]
    
    rabbitmq = _make_base_container("rabbitmq", "rabbitmq:3-management", None, "rabbitmq-diagnostics ping || exit 1", region, start_period=240)
    rabbitmq["command"] = ["sh", "-c", "if [ -n \"$RABBITMQ_ERLANG_COOKIE\" ]; then mkdir -p /var/lib/rabbitmq && echo \"$RABBITMQ_ERLANG_COOKIE\" > /var/lib/rabbitmq/.erlang.cookie && chmod 600 /var/lib/rabbitmq/.erlang.cookie && chown rabbitmq:rabbitmq /var/lib/rabbitmq/.erlang.cookie; fi && exec /usr/local/bin/docker-entrypoint.sh rabbitmq-server"]
    rabbitmq["portMappings"] = [{"containerPort": 5672, "protocol": "tcp"}, {"containerPort": 15672, "protocol": "tcp"}]
    rabbitmq["secrets"] = rabbitmq_secrets
    
    gateway_env = env_vars.copy()
    gateway_env["REDIS_URL"] = "redis://localhost:6379/0"
    gateway_env["CHROMA_URL"] = "http://localhost:8000"
    gateway_env["RABBITMQ_URL"] = "amqp://guest:guest@localhost:5672/"
    
    gateway = _make_base_container("gateway", f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/gateway:latest", 8080, "curl -f http://localhost:8080/health || exit 1", region, start_period=360)
    gateway["dependsOn"] = [{"condition": "START", "containerName": "chroma"}, {"condition": "START", "containerName": "redis"}, {"condition": "START", "containerName": "rabbitmq"}]
    gateway["secrets"] = gateway_secrets
    gateway["environment"] = build_environment_list(gateway_env)

    metrics_env = gateway_env.copy()
    metrics = _make_base_container("metrics", f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/metrics:latest", 8082, "curl -f http://localhost:8082/health || exit 1", region, start_period=180)
    metrics["dependsOn"] = [{"condition": "START", "containerName": "rabbitmq"}]
    metrics["secrets"] = gateway_secrets
    metrics["environment"] = build_environment_list(metrics_env)
    
    containers = [chroma, redis, rabbitmq, gateway, metrics]
    
    return {
        "family": "euglena-gateway",
        "containerDefinitions": containers,
        "taskRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskRole",
        "executionRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole",
        "networkMode": "awsvpc",
        "volumes": [],
        "placementConstraints": [],
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "512",
        "memory": "1024"
    }


def build_agent_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env):
    """
    Builds agent task definition.
    :param account_id: AWS account ID
    :param region: AWS region
    :param secret_name: Secrets Manager secret name
    :param secret_arn_suffix: ARN suffix for the secret
    :param secret_keys: List of secret key names
    :param env_vars: Dictionary of environment variables from .env
    :param aws_env: Dictionary of AWS environment variables from aws.env
    :returns: Task definition dictionary
    """
    secrets = build_secrets_list(secret_name, secret_arn_suffix, region, account_id, secret_keys)
    
    agent_env = env_vars.copy()
    
    gateway_service_name = aws_env.get("GATEWAY_SERVICE_NAME", "euglena-gateway")
    namespace = aws_env.get("SERVICE_DISCOVERY_NAMESPACE", "euglena.local")
    gateway_host = f"{gateway_service_name}.{namespace}"
    
    agent_env["REDIS_URL"] = f"redis://{gateway_host}:6379/0"
    agent_env["CHROMA_URL"] = f"http://{gateway_host}:8000"
    agent_env["RABBITMQ_URL"] = f"amqp://guest:guest@{gateway_host}:5672/"
    agent_env["ECS_ENABLED"] = "true"
    agent_env["AWS_REGION"] = region
    agent_env["ECS_CLUSTER"] = aws_env.get("ECS_CLUSTER", "euglena-cluster")
    
    agent = _make_base_container("agent", f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/agent:latest", 8081, "curl -f http://localhost:8081/health || exit 1", region)
    agent["secrets"] = secrets
    agent["environment"] = build_environment_list(agent_env)
    
    containers = [agent]
    
    return {
        "family": "euglena-agent",
        "containerDefinitions": containers,
        "taskRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskRole",
        "executionRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole",
        "networkMode": "awsvpc",
        "volumes": [],
        "placementConstraints": [],
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "256",
        "memory": "2048"
    }


def _get_aws_config(aws_env):
    """Gets AWS configuration from aws.env or prompts user for missing values."""
    def get_or_prompt(key, prompt_text):
        value = aws_env.get(key, "").strip() if aws_env else ""
        return value if value else input(f"{prompt_text}: ").strip()
    
    account_id = get_or_prompt("AWS_ACCOUNT_ID", "Account ID")
    region = get_or_prompt("AWS_REGION", "Region")
    secret_name = get_or_prompt("AWS_SECRET_NAME", "Secret name")
    secret_arn_suffix = get_or_prompt("AWS_SECRET_ARN_SUFFIX", "Secret ARN suffix") or ""
    
    return account_id, region, secret_name, secret_arn_suffix


def _write_task_definition(base_dir, task_def, name):
    """Write task definition JSON files."""
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


def main():
    """
    Generates ECS task definitions for gateway and agent services.
    Loads secrets from keys.env and environment variables from .env automatically.
    Expected to be run from services/ directory: python ../scripts/build-task-definition.py
    """
    base_dir = Path.cwd()
    services_dir = base_dir
    
    aws_env = load_aws_config(services_dir)
    account_id, region, secret_name, secret_arn_suffix = _get_aws_config(aws_env)
    
    if not account_id or not region:
        print("Error: Account ID and Region required.")
        return
    
    secret_keys = load_secrets_from_keys_env(services_dir)
    env_vars = load_env_variables(services_dir, secret_keys)
    
    if not secret_arn_suffix or not secret_arn_suffix.strip():
        if aws_env.get("AWS_SECRET_ARN_SUFFIX"):
            secret_arn_suffix = aws_env.get("AWS_SECRET_ARN_SUFFIX").strip()
        elif secret_keys:
            secret_input = input("Secret ARN or suffix: ").strip()
            if secret_input:
                secret_arn_suffix = extract_arn_suffix(secret_input)
    
    gateway_task_def = build_gateway_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars)
    gateway_path, gateway_redacted_path = _write_task_definition(base_dir, gateway_task_def, "gateway")
    
    agent_task_def = build_agent_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env)
    agent_path, agent_redacted_path = _write_task_definition(base_dir, agent_task_def, "agent")
    
    print(f"Generated: {gateway_path}")
    print(f"Generated: {agent_path}")
    print(f"\nRegister:")
    print(f"  aws ecs register-task-definition --cli-input-json file://{gateway_path} --region {region}")
    print(f"  aws ecs register-task-definition --cli-input-json file://{agent_path} --region {region}")
    
    from datetime import datetime
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

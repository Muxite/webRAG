"""
Builds and registers ECS task definition for single euglena service.
Loads configuration from aws.env, keys.env, and .env automatically.
"""
import json
import copy
import subprocess
import sys
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
    Loads environment variables from .env file, excluding secrets.
    :param services_dir: Services directory path where .env should be located
    :param secret_keys: List of secret key names to exclude
    :returns: Dictionary of environment variables
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


def build_environment_list(env_vars):
    """
    Converts environment variables dictionary to ECS environment list format.
    :param env_vars: Dictionary of environment variables
    :returns: List of {"name": key, "value": value} dictionaries
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


def _make_health_check(command, start_period=300, interval=60, retries=5, timeout=10):
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


def build_euglena_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env):
    """
    Builds single euglena task definition with all containers (chroma, redis, rabbitmq, agent, gateway).
    Supports EFS volumes for persistent storage if EFS file system IDs are provided in aws_env.
    
    :param account_id: AWS account ID
    :param region: AWS region
    :param secret_name: Secrets Manager secret name
    :param secret_arn_suffix: ARN suffix for the secret
    :param secret_keys: List of secret key names
    :param env_vars: Dictionary of environment variables from .env
    :param aws_env: Dictionary of AWS environment variables from aws.env (may contain EFS file system IDs)
    :returns: Task definition dictionary
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
    redis_efs_id = aws_env.get("REDIS_EFS_FILE_SYSTEM_ID", shared_efs_id).strip() if aws_env else shared_efs_id
    rabbitmq_efs_id = aws_env.get("RABBITMQ_EFS_FILE_SYSTEM_ID", shared_efs_id).strip() if aws_env else shared_efs_id
    
    chroma_health_command = "curl -f http://localhost:8000/api/v1/heartbeat || exit 1"
    chroma_mount_points = []
    if chroma_efs_id:
        chroma_mount_points.append({"sourceVolume": "chroma-data", "containerPath": "/chroma-data", "readOnly": False})
    
    chroma = {
        "cpu": 0,
        "environment": [
            {"name": "IS_PERSISTENT", "value": "TRUE"},
            {"name": "PERSIST_DIRECTORY", "value": "/chroma-data"}
        ],
        "essential": True,
        "healthCheck": _make_health_check(chroma_health_command, start_period=180),
        "image": "chromadb/chroma:latest",
        "logConfiguration": _make_log_config("chroma", region),
        "mountPoints": chroma_mount_points,
        "name": "chroma",
        "portMappings": [{"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}],
        "systemControls": [],
        "volumesFrom": []
    }
    
    redis_command = ["redis-server"]
    redis_mount_points = []
    if redis_efs_id:
        redis_command.append("--appendonly")
        redis_command.append("yes")
        redis_mount_points.append({"sourceVolume": "redis-data", "containerPath": "/data", "readOnly": False})
    
    redis = {
        "command": redis_command,
        "cpu": 0,
        "environment": [],
        "essential": True,
        "healthCheck": _make_health_check("redis-cli ping | grep PONG || exit 1", start_period=90),
        "image": "redis:7-alpine",
        "logConfiguration": _make_log_config("redis", region),
        "mountPoints": redis_mount_points,
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
        "healthCheck": _make_health_check("rabbitmq-diagnostics ping || exit 1", start_period=180),
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
            {"condition": "START", "containerName": "chroma"},
            {"condition": "START", "containerName": "redis"},
            {"condition": "START", "containerName": "rabbitmq"}
        ],
        "environment": env_list,
        "essential": True,
        "healthCheck": _make_health_check("curl -f http://localhost:8081/health || exit 1", start_period=300),
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
        "healthCheck": _make_health_check("curl -f http://localhost:8080/health || exit 1", start_period=300),
        "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/gateway:latest",
        "logConfiguration": _make_log_config("gateway", region),
        "mountPoints": [],
        "name": "gateway",
        "portMappings": [{"containerPort": 8080, "hostPort": 8080, "protocol": "tcp"}],
        "secrets": secrets,
        "systemControls": [],
        "volumesFrom": []
    }
    
    containers = [chroma, redis, rabbitmq, agent, gateway]
    
    volumes = []
    using_shared_efs = (chroma_efs_id and redis_efs_id and rabbitmq_efs_id and 
                        chroma_efs_id == redis_efs_id == rabbitmq_efs_id)
    
    if chroma_efs_id:
        efs_config = {
            "fileSystemId": chroma_efs_id,
            "transitEncryption": "ENABLED",
            "authorizationConfig": {
                "iam": "ENABLED"
            }
        }
        if using_shared_efs:
            efs_config["rootDirectory"] = "/chroma-data"
        volumes.append({
            "name": "chroma-data",
            "efsVolumeConfiguration": efs_config
        })
    
    if redis_efs_id:
        efs_config = {
            "fileSystemId": redis_efs_id,
            "transitEncryption": "ENABLED",
            "authorizationConfig": {
                "iam": "ENABLED"
            }
        }
        if using_shared_efs:
            efs_config["rootDirectory"] = "/redis-data"
        volumes.append({
            "name": "redis-data",
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
        if using_shared_efs:
            efs_config["rootDirectory"] = "/rabbitmq-data"
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
        "cpu": "1024",
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
    """
    Write task definition JSON files.
    
    :param base_dir: Base directory to write files.
    :param task_def: Task definition dictionary.
    :param name: Task definition name (euglena).
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
    
    :param task_def_path: Path to task definition JSON file (Path object).
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
        print(f"Registered task definition: {task_def_path.name}")
        return True
    else:
        print(f"Failed to register task definition: {task_def_path.name}")
        return False


def main():
    """
    Generates and registers ECS task definition for single euglena service.
    Loads secrets from keys.env and environment variables from .env automatically.
    Expected to be run from services/ directory: python ../scripts/build-task-definition.py
    """
    base_dir = Path.cwd()
    services_dir = base_dir
    
    aws_env = load_aws_config(services_dir)
    account_id, region, secret_name, secret_arn_suffix = _get_aws_config(aws_env)
    
    if not account_id or not region:
        print("Error: Account ID and Region required.")
        sys.exit(1)
    
    secret_keys = load_secrets_from_keys_env(services_dir)
    env_vars = load_env_variables(services_dir, secret_keys)
    
    if not secret_arn_suffix or not secret_arn_suffix.strip():
        if aws_env.get("AWS_SECRET_ARN_SUFFIX"):
            secret_arn_suffix = aws_env.get("AWS_SECRET_ARN_SUFFIX").strip()
        elif secret_keys:
            secret_input = input("Secret ARN or suffix: ").strip()
            if secret_input:
                secret_arn_suffix = extract_arn_suffix(secret_input)
    
    print("Generating task definition...")
    task_def = build_euglena_task_definition(account_id, region, secret_name, secret_arn_suffix, secret_keys, env_vars, aws_env)
    task_path, task_redacted_path = _write_task_definition(base_dir, task_def, "euglena")
    print(f"Generated: {task_path}")
    
    print("\nRegistering task definition with AWS ECS...")
    success = register_task_definition(task_path, region)
    
    if not success:
        print("\nError: Failed to register task definition.")
        sys.exit(1)
    
    print("\nTask definition registered successfully.")
    
    from datetime import datetime
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

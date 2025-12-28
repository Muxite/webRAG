import json
import copy
from pathlib import Path
from dotenv import dotenv_values


def load_env_files(base_dir):
    keys_env_path = base_dir / "keys.env"
    env_path = base_dir / ".env"
    keys_env = {}
    env = {}
    
    if keys_env_path.exists():
        keys_env = dict(dotenv_values(str(keys_env_path)))
        print(f"Loaded {len(keys_env)} secrets from {keys_env_path}")
    else:
        print(f"Warning: {keys_env_path} not found")
    
    if env_path.exists():
        env = dict(dotenv_values(str(env_path)))
        print(f"Loaded {len(env)} environment variables from {env_path}")
    else:
        print(f"Warning: {env_path} not found")
    
    return keys_env, env


def redact_task_definition(task_def):
    """Create a redacted version of the task definition with placeholders for account/region info."""
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
    if secret_input.startswith("arn:aws:secretsmanager:"):
        try:
            secret_idx = secret_input.find("secret:")
            if secret_idx != -1:
                after_secret = secret_input[secret_idx + 7:]
                if ":" in after_secret:
                    name_suffix_part = after_secret.split(":")[0]
                else:
                    name_suffix_part = after_secret
                if "-" in name_suffix_part:
                    last_dash_idx = name_suffix_part.rfind("-")
                    return name_suffix_part[last_dash_idx + 1:]
                return secret_input
            return secret_input
        except Exception:
            return secret_input
    return secret_input


def build_task_definition(keys_env, env, account_id, region, secret_name, secret_arn_suffix):
    env_list = [{"name": k, "value": str(v)} for k, v in sorted(env.items())]
    
    secrets = []
    if keys_env and secret_arn_suffix:
        for key in sorted(keys_env.keys()):
            secrets.append({
                "name": key,
                "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-{secret_arn_suffix}:{key}::"
            })
    
    def make_log_config(prefix):
        return {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": "/ecs/euglena",
                "awslogs-region": region,
                "awslogs-stream-prefix": prefix
            }
        }
    
    containers = [
        {
            "cpu": 0,
            "environment": [
                {"name": "IS_PERSISTENT", "value": "TRUE"},
                {"name": "PERSIST_DIRECTORY", "value": "/chroma-data"}
            ],
            "essential": True,
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/heartbeat || exit 1"],
                "interval": 30,
                "retries": 3,
                "startPeriod": 60,
                "timeout": 5
            },
            "image": "chromadb/chroma:latest",
            "logConfiguration": make_log_config("chroma"),
            "mountPoints": [],
            "name": "chroma",
            "portMappings": [{"containerPort": 8000, "hostPort": 8000, "protocol": "tcp"}],
            "systemControls": [],
            "volumesFrom": []
        },
        {
            "command": ["redis-server", "--appendonly", "yes"],
            "cpu": 0,
            "environment": [],
            "essential": True,
            "healthCheck": {
                "command": ["CMD-SHELL", "redis-cli ping | grep PONG || exit 1"],
                "interval": 30,
                "retries": 3,
                "startPeriod": 30,
                "timeout": 5
            },
            "image": "redis:7-alpine",
            "logConfiguration": make_log_config("redis"),
            "mountPoints": [],
            "name": "redis",
            "portMappings": [{"containerPort": 6379, "hostPort": 6379, "protocol": "tcp"}],
            "systemControls": [],
            "volumesFrom": []
        },
        {
            "cpu": 0,
            "environment": [],
            "essential": True,
            "healthCheck": {
                "command": ["CMD-SHELL", "rabbitmq-diagnostics ping || exit 1"],
                "interval": 30,
                "retries": 3,
                "startPeriod": 60,
                "timeout": 5
            },
            "image": "rabbitmq:3-management",
            "logConfiguration": make_log_config("rabbitmq"),
            "mountPoints": [],
            "name": "rabbitmq",
            "portMappings": [
                {"containerPort": 5672, "hostPort": 5672, "protocol": "tcp"},
                {"containerPort": 15672, "hostPort": 15672, "protocol": "tcp"}
            ],
            "secrets": [
                {
                    "name": "RABBITMQ_ERLANG_COOKIE",
                    "valueFrom": f"arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-{secret_arn_suffix}:RABBITMQ_ERLANG_COOKIE::"
                }
            ],
            "systemControls": [],
            "volumesFrom": []
        },
        {
            "cpu": 0,
            "dependsOn": [
                {"condition": "HEALTHY", "containerName": "chroma"},
                {"condition": "HEALTHY", "containerName": "redis"},
                {"condition": "HEALTHY", "containerName": "rabbitmq"}
            ],
            "environment": env_list,
            "essential": True,
            "healthCheck": {
                "command": ["CMD-SHELL", "ps aux | grep '[p]ython -m app.main' || exit 1"],
                "interval": 30,
                "retries": 3,
                "startPeriod": 120,
                "timeout": 5
            },
            "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/agent",
            "logConfiguration": make_log_config("agent"),
            "mountPoints": [],
            "name": "agent",
            "portMappings": [],
            "secrets": secrets,
            "systemControls": [],
            "volumesFrom": []
        },
        {
            "cpu": 0,
            "dependsOn": [
                {"condition": "HEALTHY", "containerName": "chroma"},
                {"condition": "HEALTHY", "containerName": "redis"},
                {"condition": "HEALTHY", "containerName": "rabbitmq"},
                {"condition": "HEALTHY", "containerName": "agent"}
            ],
            "environment": env_list,
            "essential": True,
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
                "interval": 30,
                "retries": 3,
                "startPeriod": 60,
                "timeout": 5
            },
            "image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/euglena/gateway",
            "logConfiguration": make_log_config("gateway"),
            "mountPoints": [],
            "name": "gateway",
            "portMappings": [{"containerPort": 8080, "hostPort": 8080, "protocol": "tcp"}],
            "secrets": secrets,
            "systemControls": [],
            "volumesFrom": []
        }
    ]
    
    return {
        "family": "euglena",
        "containerDefinitions": containers,
        "taskRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskRole",
        "executionRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole",
        "networkMode": "awsvpc",
        "volumes": [],
        "placementConstraints": [],
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "4096",
        "memory": "8192"
    }


def main():
    base_dir = Path.cwd()
    print(f"Working directory: {base_dir}\n")
    
    keys_env, env = load_env_files(base_dir)
    
    if not keys_env and not env:
        print("Error: No environment files found. Please ensure keys.env and/or .env exist in the current directory.")
        return
    
    print("\n" + "=" * 60)
    print("AWS Configuration")
    print("=" * 60)
    account_id = input("Enter AWS Account ID: ").strip()
    region = input("Enter AWS Region: ").strip()
    secret_name = input("Enter secret name: ").strip()
    
    if not account_id or not region or not secret_name:
        print("Error: Account ID, Region, and Secret Name are required.")
        return
    
    secret_arn_suffix = ""
    if keys_env:
        secrets_file = base_dir / "secrets.json"
        if secrets_file.exists():
            secrets_file.unlink()
        with open(secrets_file, 'w') as f:
            json.dump(keys_env, f, indent=2)
        print(f"\nCreated {secrets_file} with {len(keys_env)} secrets\n")
        
        print("To create a new secret:")
        print(f"  aws secretsmanager create-secret --name {secret_name} --secret-string file://{secrets_file} --region {region}\n")
        
        print("To update an existing secret:")
        print(f"  aws secretsmanager update-secret --secret-id {secret_name} --secret-string file://{secrets_file} --region {region}\n")
        
        print("After creating/updating the secret, AWS will return an ARN.")
        secret_input = input("Provide the full ARN or just the suffix: ").strip()
        if not secret_input:
            print("Error: Secret ARN or suffix is required.")
            return
        
        secret_arn_suffix = extract_arn_suffix(secret_input)
    
    print("\n" + "=" * 32)
    print("Generating Task Definition")
    print("=" * 32)
    
    task_def = build_task_definition(keys_env, env, account_id, region, secret_name, secret_arn_suffix)
    
    output_path = base_dir / "task_definition.json"
    if output_path.exists():
        output_path.unlink()
    with open(output_path, 'w') as f:
        json.dump(task_def, f, indent=4)
    
    redacted_def = redact_task_definition(task_def)
    redacted_path = base_dir / "task_definition.redacted.json"
    if redacted_path.exists():
        redacted_path.unlink()
    with open(redacted_path, 'w') as f:
        json.dump(redacted_def, f, indent=4)
    
    print(f"\nGenerated task definition: {output_path}")
    print(f"Generated redacted task definition: {redacted_path}")
    print(f" - {len(task_def['containerDefinitions'])} containers")
    print(f" - {len(keys_env)} secrets configured")
    print(f" - {len(env)} environment variables configured")
    print("\nYou can now register this task definition with:")
    print(f"aws ecs register-task-definition --cli-input-json file://{output_path} --region {region}")


if __name__ == "__main__":
    main()

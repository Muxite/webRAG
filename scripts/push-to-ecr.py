"""
Push Docker images to AWS ECR.

Builds and pushes images for gateway, agent, and metrics services to ECR.
Loads AWS configuration from aws.env file.

Usage:
    python scripts/push-to-ecr.py [--service gateway|agent|metrics|all]
"""
import subprocess
import sys
import os
from pathlib import Path
from dotenv import dotenv_values


def load_aws_config(services_dir):
    """
    Load AWS configuration from aws.env file.
    
    :param services_dir: Services directory path
    :return: dict with account_id, region
    """
    aws_env_path = services_dir / "aws.env"
    if not aws_env_path.exists():
        print(f"Error: {aws_env_path} not found")
        return None
    
    aws_env = dict(dotenv_values(str(aws_env_path)))
    account_id = aws_env.get("AWS_ACCOUNT_ID", "").strip()
    region = aws_env.get("AWS_REGION", "").strip()
    
    if not account_id:
        print("Error: AWS_ACCOUNT_ID not found in aws.env")
        return None
    if not region:
        print("Error: AWS_REGION not found in aws.env")
        return None
    
    return {"account_id": account_id, "region": region}


def run_command(cmd, check=True, shell=False):
    """Run a shell command and return success status."""
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    try:
        result = subprocess.run(cmd, check=check, shell=shell, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}")
        return False


def ecr_login(account_id, region):
    """Authenticate Docker with ECR."""
    registry_url = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry_url}"
    
    if not run_command(cmd, shell=True):
        print("ECR login failed")
        return False
    
    return True


def build_image(service_name, services_dir):
    """
    Build Docker image for a service.
    
    :param service_name: Service name (gateway, agent, metrics)
    :param services_dir: Path to services directory (build context)
    :return: True on success, False on failure
    """
    dockerfile_path = services_dir / service_name / ".dockerfile"
    if not dockerfile_path.exists():
        print(f"Error: Dockerfile not found: {dockerfile_path}")
        return False
    
    image_name = f"euglena/{service_name}"
    dockerfile_relative = f"{service_name}/.dockerfile"
    
    cmd = [
        "docker", "build",
        "-f", dockerfile_relative,
        "-t", image_name,
        "."
    ]
    
    original_dir = os.getcwd()
    try:
        os.chdir(services_dir)
        if not run_command(cmd):
            return False
    finally:
        os.chdir(original_dir)
    
    return True


def push_image(service_name, account_id, region):
    """Push Docker image to ECR."""
    local_image = f"euglena/{service_name}"
    registry_url = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    ecr_image = f"{registry_url}/euglena/{service_name}:latest"
    
    if not run_command(["docker", "tag", local_image, ecr_image]):
        return False
    
    if not run_command(["docker", "push", ecr_image]):
        return False
    
    return True


def ensure_ecr_repository(service_name, account_id, region):
    """Ensure ECR repository exists, create if it doesn't."""
    repository_name = f"euglena/{service_name}"
    
    check_cmd = [
        "aws", "ecr", "describe-repositories",
        "--repository-names", repository_name,
        "--region", region
    ]
    
    result = subprocess.run(check_cmd, capture_output=True)
    if result.returncode == 0:
        return True
    
    create_cmd = [
        "aws", "ecr", "create-repository",
        "--repository-name", repository_name,
        "--region", region,
        "--image-scanning-configuration", "scanOnPush=true"
    ]
    
    return run_command(create_cmd)


def main():
    """
    Main entry point.
    
    Expected to be run from services/ directory: python ../scripts/push-to-ecr.py
    """
    services_dir = Path.cwd()
    
    aws_config = load_aws_config(services_dir)
    if not aws_config:
        sys.exit(1)
    
    account_id = aws_config["account_id"]
    region = aws_config["region"]
    
    services_to_push = []
    if len(sys.argv) > 1:
        service_arg = sys.argv[1].replace("--service=", "")
        if service_arg in ["gateway", "agent", "metrics"]:
            services_to_push = [service_arg]
        elif service_arg == "all":
            services_to_push = ["gateway", "agent", "metrics"]
        else:
            print(f"Unknown service: {service_arg}")
            print("Usage: python scripts/push-to-ecr.py [--service gateway|agent|metrics|all]")
            sys.exit(1)
    else:
        services_to_push = ["gateway", "agent", "metrics"]
    
    if not ecr_login(account_id, region):
        sys.exit(1)
    
    for service in services_to_push:
        if not ensure_ecr_repository(service, account_id, region):
            continue
        
        if not build_image(service, services_dir):
            continue
        
        if not push_image(service, account_id, region):
            continue
    
    from datetime import datetime
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()


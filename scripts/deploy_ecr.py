"""
ECR operations for deployment scripts.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from deploy_common import run_command, get_image_size, load_aws_config


def push_to_ecr(services_dir: Path, aws_config: Dict, services: List[str], image_suffix: str = "") -> bool:
    """
    Push Docker images to ECR.
    :param services_dir: Services directory path.
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to push (gateway, agent).
    :param image_suffix: Suffix to add to image names.
    :returns: True when all pushes succeed, False otherwise.
    """
    account_id = aws_config["AWS_ACCOUNT_ID"]
    region = aws_config["AWS_REGION"]
    
    print("\nECR push")
    
    registry_url = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    
    print("ECR login")
    cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry_url}"
    success, _, stderr = run_command(cmd, check=False, shell=True)
    if not success:
        print(f"ECR login failed: {stderr}")
        return False
    print("ECR login ok")
    
    for service in services:
        print(f"Service: {service}")
        
        repository_name = f"euglena/{service}{image_suffix}"
        
        print("Repository check")
        check_cmd = ["aws", "ecr", "describe-repositories", "--repository-names", repository_name, "--region", region]
        success, _, _ = run_command(check_cmd, check=False, capture=True)
        if not success:
            create_cmd = ["aws", "ecr", "create-repository", "--repository-name", repository_name, "--region", region, "--image-scanning-configuration", "scanOnPush=true"]
            success, _, stderr = run_command(create_cmd, check=False, capture=True)
            if not success:
                print(f"Repository create failed: {stderr}")
                continue
            print("Repository created")
        else:
            print("Repository exists")
        
        dockerfile_path = services_dir / service / ".dockerfile"
        if not dockerfile_path.exists():
            print(f"Dockerfile missing: {dockerfile_path}")
            continue
        
        print("Build image")
        image_name = f"euglena/{service}{image_suffix}"
        dockerfile_relative = f"{service}/.dockerfile"
        build_cmd = ["docker", "build", "-f", dockerfile_relative, "-t", image_name, "."]
        
        original_dir = os.getcwd()
        try:
            os.chdir(services_dir)
            process = subprocess.Popen(
                build_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            for line in process.stdout:
                try:
                    safe_line = line.rstrip().encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace')
                    print(safe_line)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    safe_line = line.rstrip().encode('ascii', errors='replace').decode('ascii')
                    print(safe_line)
            process.wait()
            if process.returncode != 0:
                print(f"Build failed: exit {process.returncode}")
                continue
        finally:
            os.chdir(original_dir)
        
        image_size = get_image_size(image_name)
        if image_size:
            print(f"Image built: {image_size}")
        else:
            print("Image built")
        
        print("Tag and push")
        ecr_image = f"{registry_url}/euglena/{service}{image_suffix}:latest"
        success, _, stderr = run_command(["docker", "tag", image_name, ecr_image], check=False, capture=True)
        if not success:
            print(f"Tag failed: {stderr}")
            continue
        
        print("Push")
        push_cmd = ["docker", "push", ecr_image]
        process = subprocess.Popen(
            push_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=1
        )
        if process.stdout:
            for line in iter(process.stdout.readline, b""):
                try:
                    decoded = line.decode(sys.stdout.encoding or "utf-8", errors="replace").rstrip()
                    print(decoded)
                except Exception:
                    print(line.decode("utf-8", errors="replace").rstrip())
        process.wait()
        if process.returncode != 0:
            print(f"Push failed: exit {process.returncode}")
            continue
        print("Image pushed")
    
    return True


def parse_args():
    """
    Parse CLI arguments.
    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Push images to ECR")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env")
    parser.add_argument("--services", default="gateway,agent,metrics",
                       help="Comma-separated service list (default: gateway,agent,metrics)")
    parser.add_argument("--suffix", default="", help="Optional image suffix")
    return parser.parse_args()


def main():
    """
    Run the ECR push workflow.
    :returns: None.
    """
    args = parse_args()
    services_dir = args.services_dir or Path.cwd()
    aws_config = load_aws_config(services_dir)
    services = [s.strip() for s in args.services.split(",") if s.strip()]
    success = push_to_ecr(services_dir, aws_config, services, image_suffix=args.suffix)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

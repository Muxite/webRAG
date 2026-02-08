"""
ECR operations for deployment scripts.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from deploy_common import run_command, get_image_size


def push_to_ecr(services_dir: Path, aws_config: Dict, services: List[str], image_suffix: str = "") -> bool:
    """
    Push Docker images to ECR.
    
    :param services_dir: Services directory path.
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to push (gateway, agent).
    :param image_suffix: Suffix to add to image names (e.g., "-autoscale").
    :returns: True on success.
    """
    account_id = aws_config["AWS_ACCOUNT_ID"]
    region = aws_config["AWS_REGION"]
    
    print("\n=== Pushing Images to ECR ===")
    
    registry_url = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    
    print("Authenticating with ECR...")
    cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry_url}"
    success, _, stderr = run_command(cmd, check=False, shell=True)
    if not success:
        print(f"  FAIL: ECR login failed: {stderr}")
        return False
    print("  OK: ECR login successful")
    
    for service in services:
        print(f"Processing {service}...")
        
        repository_name = f"euglena/{service}{image_suffix}"
        
        print("  Checking repository...")
        check_cmd = ["aws", "ecr", "describe-repositories", "--repository-names", repository_name, "--region", region]
        success, _, _ = run_command(check_cmd, check=False, capture=True)
        if not success:
            create_cmd = ["aws", "ecr", "create-repository", "--repository-name", repository_name, "--region", region, "--image-scanning-configuration", "scanOnPush=true"]
            success, _, stderr = run_command(create_cmd, check=False, capture=True)
            if not success:
                print(f"    FAIL: Failed to create repository: {stderr}")
                continue
            print(f"    OK: Created repository")
        else:
            print(f"    OK: Repository exists")
        
        dockerfile_path = services_dir / service / ".dockerfile"
        if not dockerfile_path.exists():
            print(f"    FAIL: Dockerfile not found: {dockerfile_path}")
            continue
        
        print("  Building image...")
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
                print(f"    FAIL: Build failed with exit code {process.returncode}")
                continue
        finally:
            os.chdir(original_dir)
        
        image_size = get_image_size(image_name)
        if image_size:
            print(f"    OK: Image built (size: {image_size})")
        else:
            print(f"    OK: Image built")
        
        print(f"  Tagging and pushing...")
        ecr_image = f"{registry_url}/euglena/{service}{image_suffix}:latest"
        success, _, stderr = run_command(["docker", "tag", image_name, ecr_image], check=False, capture=True)
        if not success:
            print(f"    FAIL: Tag failed: {stderr}")
            continue
        
        print("  Pushing to ECR (this may take a while)...")
        push_cmd = ["docker", "push", ecr_image]
        process = subprocess.Popen(
            push_cmd,
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
            print(f"    FAIL: Push failed with exit code {process.returncode}")
            continue
        print(f"    OK: Image pushed to ECR")
    
    return True

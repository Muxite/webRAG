"""
Deployment script for Euglena services.

Works from scratch (creates everything) or updates existing deployment.
Uses OOP modules for network discovery and ECS infrastructure management.
"""
import boto3
import subprocess
import sys
import time
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import dotenv_values

try:
    from scripts.network_discovery import NetworkDiscovery
    from scripts.ecs_infrastructure import EcsInfrastructure
    from scripts.network_utils import validate_network_configuration, fix_security_group_rules
except ImportError:
    from network_discovery import NetworkDiscovery
    from ecs_infrastructure import EcsInfrastructure
    from network_utils import validate_network_configuration, fix_security_group_rules


def load_aws_config(services_dir: Path) -> Dict:
    """
    Load AWS configuration from aws.env file.
    
    :param services_dir: Services directory path.
    :returns: Configuration dictionary.
    """
    aws_env_path = services_dir / "aws.env"
    if not aws_env_path.exists():
        print(f"Error: {aws_env_path} not found")
        sys.exit(1)
    
    return dict(dotenv_values(str(aws_env_path)))


def run_command(cmd: List[str], check: bool = True, capture: bool = False, shell: bool = False) -> Tuple[bool, str, str]:
    """
    Run a shell command and return result.
    
    :param cmd: Command as list of strings or string.
    :param check: Whether to raise on non-zero exit.
    :param capture: Whether to capture output.
    :param shell: Whether to run in shell.
    :returns: Tuple of (success, stdout, stderr).
    """
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            encoding='utf-8',
            errors='replace',
            shell=shell
        )
        return True, result.stdout if capture else "", result.stderr if capture else ""
    except subprocess.CalledProcessError as e:
        return False, e.stdout if capture else "", e.stderr if capture else str(e)
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}"


def push_to_ecr(services_dir: Path, aws_config: Dict, services: List[str]) -> bool:
    """
    Push Docker images to ECR.
    
    :param services_dir: Services directory path.
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to push (gateway, agent).
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
        
        repository_name = f"euglena/{service}"
        
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
        image_name = f"euglena/{service}"
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
                print(line.rstrip())
            process.wait()
            if process.returncode != 0:
                print(f"    FAIL: Build failed with exit code {process.returncode}")
                continue
        finally:
            os.chdir(original_dir)
        print(f"    OK: Image built")
        
        print(f"  Tagging and pushing...")
        ecr_image = f"{registry_url}/euglena/{service}:latest"
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
            print(line.rstrip())
        process.wait()
        if process.returncode != 0:
            print(f"    FAIL: Push failed with exit code {process.returncode}")
            continue
        print(f"    OK: Image pushed to ECR")
    
    return True


def build_and_register_task_definitions(services_dir: Path) -> bool:
    """
    Build and register task definitions.
    
    :param services_dir: Services directory path.
    :returns: True on success.
    """
    print("\n=== Building Task Definitions ===")
    
    script_path = services_dir.parent / "scripts" / "build-task-definition.py"
    process = subprocess.Popen(
        ["python", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )
    for line in process.stdout:
        print(line.rstrip())
    process.wait()
    
    if process.returncode != 0:
        print(f"  FAIL: Failed to build task definitions (exit code {process.returncode})")
        return False
    
    print("  OK: Task definitions built and registered")
    return True


def get_network_config(aws_config: Dict, network_discovery: NetworkDiscovery) -> Optional[Dict]:
    """
    Get network configuration for ECS service.
    
    :param aws_config: AWS configuration dictionary.
    :param network_discovery: NetworkDiscovery instance.
    :returns: Network configuration dictionary or None on error.
    """
    vpc_id = aws_config.get("VPC_ID")
    subnet_ids = aws_config.get("SUBNET_IDS", "").split(",") if aws_config.get("SUBNET_IDS") else []
    security_group_ids = aws_config.get("SECURITY_GROUP_IDS", "").split(",") if aws_config.get("SECURITY_GROUP_IDS") else []
    
    vpc = network_discovery.find_vpc()
    if not vpc:
        print("  FAIL: Could not find VPC")
        return None
    
    vpc_id = vpc["VpcId"]
    
    if not subnet_ids:
        subnets = network_discovery.find_subnets(vpc_id)
        if not subnets:
            print("  FAIL: Could not find subnets")
            return None
        subnet_ids = [s["SubnetId"] for s in subnets[:2]]
    
    if not security_group_ids:
        sgs = network_discovery.find_security_groups(vpc_id)
        if not sgs:
            print("  FAIL: Could not find security groups")
            return None
        security_group_ids = [sg["GroupId"] for sg in sgs[:1]]
    
    assign_public_ip = aws_config.get("ASSIGN_PUBLIC_IP", "ENABLED").upper()
    
    return {
        "subnets": subnet_ids,
        "securityGroups": security_group_ids,
        "assignPublicIp": assign_public_ip
    }


def wait_for_services_stable(aws_config: Dict, services: List[str], timeout: int = 600):
    """
    Wait for services to become stable.
    
    :param aws_config: AWS configuration dictionary.
    :param services: List of services to wait for (service names as-is).
    :param timeout: Maximum wait time in seconds.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print("\n=== Waiting for Services to Stabilize ===")
    
    for service_name in services:
        print(f"\n  Waiting for {service_name}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = ecs_client.describe_services(cluster=cluster, services=[service_name])
                services_list = response.get("services", [])
                
                if services_list:
                    svc = services_list[0]
                    deployments = svc.get("deployments", [])
                    primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
                    
                    if primary:
                        running = primary.get("runningCount", 0)
                        desired = primary.get("desiredCount", 0)
                        pending = primary.get("pendingCount", 0)
                        
                        if running == desired and pending == 0:
                            print(f"    OK: {service_name} is stable ({running}/{desired} running)")
                            break
                        
                        print(f"    {service_name}: {running}/{desired} running, {pending} pending")
                
                time.sleep(10)
            except Exception as e:
                print(f"    Error checking service: {e}")
                time.sleep(10)
        else:
            print(f"    WARN: {service_name} did not stabilize within {timeout}s")


def main():
    """
    Main deployment entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=["euglena", "all"], default="all",
                       help="Service to deploy (default: all, deploys single euglena service)")
    parser.add_argument("--skip-ecr", action="store_true",
                       help="Skip pushing images to ECR")
    parser.add_argument("--skip-network-check", action="store_true",
                       help="Skip network validation")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for services to stabilize after deployment")
    
    args = parser.parse_args()
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print("Error: Must run from services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    print("=" * 60)
    print("Euglena Deployment")
    print("=" * 60)
    print(f"Service: euglena (single service with all containers)")
    print(f"Region: {aws_config['AWS_REGION']}")
    print(f"Cluster: {aws_config['ECS_CLUSTER']}")
    print("=" * 60)
    
    all_success = True
    
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    
    network_discovery = NetworkDiscovery(region=region, vpc_id=aws_config.get("VPC_ID"))
    ecs_infrastructure = EcsInfrastructure(region=region, cluster_name=cluster)
    
    if not ecs_infrastructure.ensure_cluster():
        print("\nFAIL: Cluster setup failed")
        all_success = False
    
    if not args.skip_ecr:
        ecr_services = ["gateway", "agent"]
        if not push_to_ecr(services_dir, aws_config, ecr_services):
            print("\nFAIL: ECR push failed")
            all_success = False
    else:
        print("\nSKIP: Skipping ECR push")
    
    if not build_and_register_task_definitions(services_dir):
        print("\nFAIL: Task definition build failed")
        all_success = False
    
    if not args.skip_network_check:
        is_valid, issues = validate_network_configuration(aws_config)
        if not is_valid:
            print("\nWARN: Network validation had issues")
            print("Attempting to fix security group rules...")
            success, messages = fix_security_group_rules(aws_config)
            if success:
                print("  OK: Security group rules fixed")
                time.sleep(5)
                is_valid, issues = validate_network_configuration(aws_config)
                if not is_valid:
                    print(f"  WARN: Network still has {len(issues)} issues after fix")
            else:
                print(f"  WARN: Failed to fix: {messages}")
    else:
        print("\nSKIP: Skipping network validation")
    
    print("\n=== Fixing IAM Permissions ===")
    try:
        import importlib.util
        script_dir = Path(__file__).parent
        iam_script_path = script_dir / "fix-iam-permissions.py"
        if iam_script_path.exists():
            spec = importlib.util.spec_from_file_location("fix_iam_permissions", iam_script_path)
            fix_iam_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fix_iam_module)
            if hasattr(fix_iam_module, 'add_task_protection_permission'):
                if not fix_iam_module.add_task_protection_permission(aws_config):
                    print("  WARN: IAM permission fix had issues (continuing anyway)")
            else:
                print("  WARN: add_task_protection_permission function not found (skipping)")
        else:
            print("  WARN: fix-iam-permissions.py not found (skipping)")
    except Exception as e:
        print(f"  WARN: Could not fix IAM permissions: {e} (continuing anyway)")
    
    network_config = get_network_config(aws_config, network_discovery)
    if not network_config:
        print("\nFAIL: Could not determine network configuration")
        all_success = False
    else:
        print("\n=== Setting Up Load Balancer ===")
        load_balancers = None
        target_group_name = aws_config.get("TARGET_GROUP_NAME")
        if target_group_name:
            try:
                elbv2_client = boto3.client("elbv2", region_name=region)
                response = elbv2_client.describe_target_groups(Names=[target_group_name])
                target_groups = response.get("TargetGroups", [])
                if target_groups:
                    target_group_arn = target_groups[0]["TargetGroupArn"]
                    container_name = "gateway"
                    container_port = 8080
                    load_balancers = [{
                        "targetGroupArn": target_group_arn,
                        "containerName": container_name,
                        "containerPort": container_port
                    }]
                    print(f"  OK: Target group {target_group_name} found: {target_group_arn}")
                else:
                    print(f"  WARN: Target group {target_group_name} not found (continuing without ALB)")
            except Exception as e:
                print(f"  WARN: Could not configure load balancer: {e} (continuing without ALB)")
        else:
            print("  SKIP: TARGET_GROUP_NAME not set in aws.env (skipping ALB configuration)")
        
        print("\n=== Creating/Updating ECS Service ===")
        service_name = "euglena-service"
        task_family = "euglena"
        
        if not ecs_infrastructure.ensure_service(
            service_name=service_name,
            task_family=task_family,
            network_config=network_config,
            desired_count=1,
            load_balancers=load_balancers,
            health_check_grace_period=100,
            enable_az_rebalancing=True
        ):
            print(f"  FAIL: Failed to create/update {service_name}")
            all_success = False
    
    if args.wait:
        time.sleep(120)
        wait_for_services_stable(aws_config, ["euglena-service"])
    
    if all_success:
        print("\n" + "=" * 60)
        print("OK: Deployment complete")
        print("=" * 60)
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("FAIL: Deployment completed with errors")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
    
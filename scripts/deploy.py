"""
Deployment script for Euglena services.
Supports single service or autoscale gateway/agent deployment modes.
"""
import boto3
import subprocess
import sys
import time
import argparse
import os
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import dotenv_values

try:
    from scripts.network_discovery import NetworkDiscovery
    from scripts.ecs_infrastructure import EcsInfrastructure
    from scripts.network_utils import validate_network_configuration, fix_security_group_rules
    from scripts.efs_manager import EfsManager
    from scripts.deployment_mode import DeploymentMode
except ImportError:
    from network_discovery import NetworkDiscovery
    from ecs_infrastructure import EcsInfrastructure
    from network_utils import validate_network_configuration, fix_security_group_rules
    from efs_manager import EfsManager
    from deployment_mode import DeploymentMode


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


def get_image_size(image_name: str) -> Optional[str]:
    """
    Get the size of a Docker image in human-readable format.
    
    :param image_name: Docker image name (e.g., "euglena/agent").
    :returns: Formatted size string (e.g., "245.3 MB") or None on error.
    """
    try:
        cmd = ["docker", "inspect", "--format", "{{.Size}}", image_name]
        success, stdout, stderr = run_command(cmd, check=False, capture=True)
        if not success or not stdout.strip():
            return None
        
        size_bytes = int(stdout.strip())
        
        if size_bytes >= 1024 * 1024 * 1024:
            size_gb = size_bytes / (1024 * 1024 * 1024)
            return f"{size_gb:.2f} GB"
        elif size_bytes >= 1024 * 1024:
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f} MB"
        elif size_bytes >= 1024:
            size_kb = size_bytes / 1024
            return f"{size_kb:.2f} KB"
        else:
            return f"{size_bytes} B"
    except (ValueError, Exception):
        return None


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


def build_and_register_task_definitions(services_dir: Path, mode: DeploymentMode = DeploymentMode.SINGLE) -> bool:
    """
    Build and register task definitions.
    
    :param services_dir: Services directory path.
    :param mode: Deployment mode enum.
    :returns: True on success.
    """
    print("\n=== Building Task Definitions ===")
    
    script_path = services_dir.parent / "scripts" / "build_task_definition.py"
    cmd = ["python", str(script_path), "--mode", mode.value]
    process = subprocess.Popen(
        cmd,
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
        "vpcId": vpc_id,
        "subnets": subnet_ids,
        "securityGroups": security_group_ids,
        "assignPublicIp": assign_public_ip
    }


def stop_old_tasks(aws_config: Dict, service_name: str) -> bool:
    """
    Stop all running tasks for a service before deployment to ensure clean state.
    
    :param aws_config: AWS configuration dictionary.
    :param service_name: Service name.
    :returns: True on success, False on error.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print(f"\n=== Stopping Old Tasks for {service_name} ===")
    
    try:
        response = ecs_client.list_tasks(
            cluster=cluster,
            serviceName=service_name,
            desiredStatus="RUNNING"
        )
        task_arns = response.get("taskArns", [])
        
        if not task_arns:
            print(f"  OK: No running tasks to stop")
            return True
        
        print(f"  Found {len(task_arns)} running task(s), stopping...")
        
        for task_arn in task_arns:
            try:
                ecs_client.stop_task(
                    cluster=cluster,
                    task=task_arn,
                    reason="Stopping for clean deployment"
                )
                print(f"    Stopped task: {task_arn.split('/')[-1]}")
            except Exception as e:
                print(f"    WARN: Failed to stop task {task_arn.split('/')[-1]}: {e}")
        
        max_wait = 60
        wait_interval = 5
        waited = 0
        
        while waited < max_wait:
            response = ecs_client.list_tasks(
                cluster=cluster,
                serviceName=service_name,
                desiredStatus="RUNNING"
            )
            remaining = len(response.get("taskArns", []))
            
            if remaining == 0:
                print(f"  OK: All tasks stopped")
                return True
            
            print(f"    Waiting for {remaining} task(s) to stop... ({waited}s)")
            time.sleep(wait_interval)
            waited += wait_interval
        
        if remaining > 0:
            print(f"  WARN: {remaining} task(s) still running after {max_wait}s (continuing anyway)")
        
        return True
    except Exception as e:
        error_str = str(e)
        if "ServiceNotFoundException" in error_str or "does not exist" in error_str:
            print(f"  OK: Service does not exist yet (no tasks to stop)")
            return True
        print(f"  WARN: Error stopping old tasks: {e} (continuing anyway)")
        return True


def stop_other_mode_services(aws_config: Dict, current_mode: DeploymentMode) -> bool:
    """
    Stop services from the other deployment mode to prevent conflicts.
    
    :param aws_config: AWS configuration dictionary.
    :param current_mode: Current deployment mode enum.
    :returns: True on success.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    if current_mode == DeploymentMode.SINGLE:
        other_services = ["euglena-gateway", "euglena-agent"]
        print("\n=== Stopping Autoscale Services ===")
    else:
        other_services = ["euglena-service"]
        print("\n=== Stopping Single Service ===")
    
    for service_name in other_services:
        try:
            response = ecs_client.describe_services(
                cluster=cluster,
                services=[service_name]
            )
            services = response.get("services", [])
            
            if not services:
                print(f"  OK: {service_name} does not exist (nothing to stop)")
                continue
            
            service = services[0]
            current_desired = service.get("desiredCount", 0)
            
            if current_desired == 0:
                print(f"  OK: {service_name} already stopped (desired count: 0)")
                continue
            
            print(f"  Stopping {service_name} (current desired count: {current_desired})...")
            
            ecs_client.update_service(
                cluster=cluster,
                service=service_name,
                desiredCount=0
            )
            
            print(f"  OK: Set {service_name} desired count to 0")
            
            max_wait = 120
            wait_interval = 5
            waited = 0
            
            while waited < max_wait:
                response = ecs_client.describe_services(
                    cluster=cluster,
                    services=[service_name]
                )
                services = response.get("services", [])
                if services:
                    service = services[0]
                    running = service.get("runningCount", 0)
                    if running == 0:
                        print(f"  OK: {service_name} stopped (all tasks terminated)")
                        break
                    print(f"    Waiting for {service_name} to stop... ({running} tasks running, {waited}s)")
                time.sleep(wait_interval)
                waited += wait_interval
            
            if waited >= max_wait:
                print(f"  WARN: {service_name} did not fully stop within {max_wait}s (continuing anyway)")
        
        except Exception as e:
            error_str = str(e)
            if "ServiceNotFoundException" in error_str or "does not exist" in error_str:
                print(f"  OK: {service_name} does not exist (nothing to stop)")
            else:
                print(f"  WARN: Error stopping {service_name}: {e} (continuing anyway)")
    
    return True


def cleanup_old_deployments(aws_config: Dict, service_name: str) -> bool:
    """
    Clean up old non-primary deployments to ensure only the current deployment exists.
    
    :param aws_config: AWS configuration dictionary.
    :param service_name: Service name.
    :returns: True on success, False on error.
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    ecs_client = boto3.client("ecs", region_name=region)
    
    print(f"\n=== Cleaning Up Old Deployments for {service_name} ===")
    
    try:
        response = ecs_client.describe_services(
            cluster=cluster,
            services=[service_name]
        )
        services = response.get("services", [])
        
        if not services:
            print(f"  OK: Service does not exist (no deployments to clean)")
            return True
        
        service = services[0]
        deployments = service.get("deployments", [])
        
        primary = next((d for d in deployments if d.get("status") == "PRIMARY"), None)
        if not primary:
            print(f"  WARN: No primary deployment found")
            return True
        
        non_primary = [d for d in deployments if d.get("id") != primary.get("id")]
        
        if not non_primary:
            print(f"  OK: No old deployments to clean (only primary exists)")
            return True
        
        print(f"  Found {len(non_primary)} old deployment(s)")
        
        for deployment in non_primary:
            dep_id = deployment.get("id", "unknown")
            status = deployment.get("status", "UNKNOWN")
            running = deployment.get("runningCount", 0)
            print(f"    Old deployment {dep_id}: status={status}, running={running}")
        
        print(f"  OK: Old deployments will be cleaned up automatically by ECS")
        return True
    except Exception as e:
        error_str = str(e)
        if "ServiceNotFoundException" in error_str:
            print(f"  OK: Service does not exist (no deployments to clean)")
            return True
        print(f"  WARN: Error checking deployments: {e} (continuing anyway)")
        return True


def ensure_exact_service_config(ecs_infrastructure: EcsInfrastructure, aws_config: Dict,
                                service_name: str, task_family: str, network_config: Dict,
                                load_balancers: Optional[List[Dict]], desired_count: int,
                                force_deploy: bool = False) -> bool:
    """
    Ensure service has exactly the specified configuration, removing any leftover settings.
    
    This function ensures a clean deployment by:
    1. Stopping old running tasks
    2. Cleaning up old deployments
    3. Updating service with exact configuration (no leftover settings)
    
    :param ecs_infrastructure: EcsInfrastructure instance.
    :param aws_config: AWS configuration dictionary.
    :param service_name: Service name.
    :param task_family: Task definition family.
    :param network_config: Network configuration.
    :param load_balancers: Load balancer configuration (None to remove old config).
    :param desired_count: Desired task count.
    :param force_deploy: Force a new deployment even when task definition is unchanged.
    :returns: True on success.
    """
    print(f"\n=== Ensuring Exact Service Configuration ===")
    
    if not stop_old_tasks(aws_config, service_name):
        print("  WARN: Failed to stop old tasks (continuing anyway)")
    
    cleanup_old_deployments(aws_config, service_name)
    
    return ecs_infrastructure.ensure_service(
        service_name=service_name,
        task_family=task_family,
        network_config=network_config,
        desired_count=desired_count,
        load_balancers=load_balancers,
        service_registries=None,
        health_check_grace_period=100,
        enable_az_rebalancing=True,
        force_deploy=force_deploy
    )


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


def parse_args():
    """
    Parse CLI arguments.

    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "autoscale"], default="single",
                       help="Deployment mode: single (all containers) or autoscale (gateway/agent)")
    parser.add_argument("--skip-ecr", action="store_true",
                       help="Skip building and pushing images to ECR (use existing images)")
    parser.add_argument("--skip-network-check", action="store_true",
                       help="Skip network validation")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for services to stabilize after deployment")
    parser.add_argument("--force-deploy", action="store_true",
                       help="Force new deployment even if task definition is unchanged")
    parser.add_argument("--update-secrets", action="store_true",
                       help="Update Secrets Manager values from keys.env before deployment")
    
    return parser.parse_args()

def main():
    """Main deployment entry point."""
    args = parse_args()
    mode = DeploymentMode.from_string(args.mode)
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print("Error: Must run from services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    print("=" * 60)
    print("Euglena Deployment")
    print("=" * 60)
    if mode == DeploymentMode.SINGLE:
        print(f"Mode: Single service (all containers)")
    else:
        print(f"Mode: Autoscale services (gateway + agent)")
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
    
    if args.update_secrets:
        print("\n=== Updating Secrets Manager ===")
        try:
            import sys as _sys
            script_dir = Path(__file__).parent
            if str(script_dir) not in _sys.path:
                _sys.path.insert(0, str(script_dir))
            from register_secrets import update_secrets_from_keys_env
        except Exception as e:
            print(f"  WARN: Secrets update failed: {e}")
            all_success = False
        else:
            if update_secrets_from_keys_env(services_dir):
                print("  OK: Secrets updated")
                args.force_deploy = True
            else:
                print("  WARN: Secrets update failed")
                all_success = False

    if not args.skip_ecr:
        ecr_services = ["gateway", "agent"]
        if not push_to_ecr(services_dir, aws_config, ecr_services):
            print("\nFAIL: ECR push failed")
            all_success = False
    else:
        print("\nSKIP: Skipping ECR push")
    
    if not build_and_register_task_definitions(services_dir, mode):
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
        iam_script_path = script_dir / "fix_iam_permissions.py"
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
            print("  WARN: fix_iam_permissions.py not found (skipping)")
    except Exception as e:
        print(f"  WARN: Could not fix IAM permissions: {e} (continuing anyway)")
    
    print("\n=== Setting Up EFS Mount Targets ===")
    efs_file_system_id = aws_config.get("EFS_FILE_SYSTEM_ID", "").strip()
    if efs_file_system_id:
        try:
            efs_manager = EfsManager(region=region, file_system_id=efs_file_system_id)
            fs = efs_manager.get_file_system()
            if fs:
                print(f"  OK: EFS file system found: {efs_file_system_id}")
                
                network_config = get_network_config(aws_config, network_discovery)
                if not network_config:
                    print("  WARN: Could not determine network configuration (skipping mount target setup)")
                else:
                    vpc_id = network_config.get("vpcId") or aws_config.get("VPC_ID")
                    subnet_ids = network_config.get("subnets", [])
                    security_group_ids = aws_config.get("SECURITY_GROUP_IDS", "").split(",") if aws_config.get("SECURITY_GROUP_IDS") else []
                    ecs_security_group_ids = network_config.get("securityGroups", [])
                    
                    if subnet_ids:
                        if not security_group_ids and vpc_id:
                            sg_id = efs_manager.get_or_create_security_group_for_efs(
                                vpc_id, 
                                ecs_security_group_ids=ecs_security_group_ids
                            )
                            if sg_id:
                                security_group_ids = [sg_id]
                        
                        mount_targets = efs_manager.ensure_mount_targets(
                            subnet_ids=[s.strip() for s in subnet_ids if s.strip()],
                            security_group_ids=[s.strip() for s in security_group_ids if s.strip()]
                        )
                        print(f"  OK: {len(mount_targets)} mount target(s) ready")
                        
                        if ecs_security_group_ids:
                            actual_efs_sg_ids = efs_manager.get_mount_target_security_groups()
                            if not actual_efs_sg_ids and security_group_ids:
                                actual_efs_sg_ids = security_group_ids
                            
                            for efs_sg_id in actual_efs_sg_ids:
                                print(f"  Ensuring ECS access to EFS security group {efs_sg_id}...")
                                try:
                                    efs_manager._ensure_ecs_access_to_efs(efs_sg_id, ecs_security_group_ids)
                                except Exception as sg_error:
                                    print(f"  WARN: Could not update security group {efs_sg_id}: {sg_error}")
                                    print(f"  NOTE: You may need to manually add NFS (port 2049) access from ECS security groups")
                            print(f"  OK: EFS security groups configured for ECS access")
                    else:
                        print("  WARN: No subnet IDs configured (skipping mount target setup)")
            else:
                print(f"  WARN: EFS file system {efs_file_system_id} not found (skipping)")
        except Exception as e:
            print(f"  WARN: Could not set up EFS mount targets: {e} (continuing anyway)")
            print(f"  NOTE: EFS volumes will be skipped if security groups are not configured correctly")
    else:
        print("  SKIP: EFS_FILE_SYSTEM_ID not set in aws.env (EFS volumes will not be used)")
    
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
        
        print("\n=== Stopping Other Mode Services ===")
        stop_other_mode_services(aws_config, mode)
        
        print("\n=== Creating/Updating ECS Service ===")
        
        if mode == DeploymentMode.SINGLE:
            service_name = "euglena-service"
            task_family = "euglena"
            
            if not ensure_exact_service_config(
                ecs_infrastructure=ecs_infrastructure,
                aws_config=aws_config,
                service_name=service_name,
                task_family=task_family,
                network_config=network_config,
                load_balancers=load_balancers,
                desired_count=1,
                force_deploy=args.force_deploy
            ):
                print(f"  FAIL: Failed to create/update {service_name}")
                all_success = False
            
        if args.wait:
            time.sleep(120)
            wait_for_services_stable(aws_config, ["euglena-service"])
        else:
            gateway_service_name = "euglena-gateway"
            agent_service_name = "euglena-agent"
            gateway_task_family = "euglena-gateway"
            agent_task_family = "euglena-agent"
            
            if not ensure_exact_service_config(
                ecs_infrastructure=ecs_infrastructure,
                aws_config=aws_config,
                service_name=gateway_service_name,
                task_family=gateway_task_family,
                network_config=network_config,
                load_balancers=load_balancers,
                desired_count=1,
                force_deploy=args.force_deploy
            ):
                print(f"  FAIL: Failed to create/update {gateway_service_name}")
                all_success = False
            
            if not ensure_exact_service_config(
                ecs_infrastructure=ecs_infrastructure,
                aws_config=aws_config,
                service_name=agent_service_name,
                task_family=agent_task_family,
                network_config=network_config,
                load_balancers=None,
                desired_count=1,
                force_deploy=args.force_deploy
            ):
                print(f"  FAIL: Failed to create/update {agent_service_name}")
                all_success = False
            
        if args.wait:
            time.sleep(120)
            wait_for_services_stable(aws_config, [gateway_service_name, agent_service_name])
    
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
    


"""
Deployment script for autoscale mode.
Deploys separate gateway and agent services.
"""
import sys
import time
import argparse
import os
import boto3
from pathlib import Path
from datetime import datetime

from deploy_common import load_aws_config
from deploy_ecr import push_to_ecr
from deploy_task_definitions import build_and_register_task_definitions
from deploy_network import get_network_config
from deploy_ecs import ensure_exact_service_config, wait_for_services_stable, stop_other_mode_services
from deploy_service_discovery import setup_service_discovery

try:
    from scripts.deployment_mode import DeploymentMode
except ImportError:
    from deployment_mode import DeploymentMode

try:
    from scripts.network_discovery import NetworkDiscovery
    from scripts.ecs_infrastructure import EcsInfrastructure
    from scripts.network_utils import validate_network_configuration, fix_security_group_rules
    from scripts.efs_manager import EfsManager
except ImportError:
    from network_discovery import NetworkDiscovery
    from ecs_infrastructure import EcsInfrastructure
    from network_utils import validate_network_configuration, fix_security_group_rules
    from efs_manager import EfsManager


def parse_args():
    """
    Parse CLI arguments.

    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Deploy autoscale mode")
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


def resolve_deployment_number(region: str, repository_name: str) -> str:
    """
    Resolve deployment number from ECR image history count.
    :param region: AWS region.
    :param repository_name: ECR repository name.
    :returns: Deployment number string.
    """
    try:
        client = boto3.client("ecr", region_name=region)
        paginator = client.get_paginator("describe_images")
        count = 0
        for page in paginator.paginate(repositoryName=repository_name):
            count += len(page.get("imageDetails", []))
        return str(count)
    except Exception as e:
        print(f"  WARN: Could not resolve deployment number: {e}")
        return "0"

def main():
    """Main deployment entry point for autoscale mode."""
    args = parse_args()
    
    services_dir = Path.cwd()
    if not (services_dir / "aws.env").exists():
        print("Error: Must run from services/ directory")
        sys.exit(1)
    
    aws_config = load_aws_config(services_dir)
    
    print("=" * 60)
    print("Euglena Deployment - Autoscale Mode")
    print("=" * 60)
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
            import subprocess
            script_dir = Path(__file__).parent
            secrets_cmd = [sys.executable, str(script_dir / "update_secrets.py")]
            result = subprocess.run(secrets_cmd, cwd=services_dir, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  WARN: Secrets update failed: {result.stderr.strip() or result.stdout.strip()}")
                all_success = False
            else:
                print("  OK: Secrets updated")
                args.force_deploy = True
        except Exception as e:
            print(f"  WARN: Secrets update failed: {e}")
            all_success = False

    if not args.skip_ecr:
        ecr_services = ["gateway", "agent", "metrics"]
        if not push_to_ecr(services_dir, aws_config, ecr_services, image_suffix=""):
            print("\nFAIL: ECR push failed")
            all_success = False
    else:
        print("\nSKIP: Skipping ECR push")
    
    deployment_number = resolve_deployment_number(region, "euglena/gateway")
    os.environ["DEPLOYMENT_NUMBER"] = deployment_number

    if not build_and_register_task_definitions(services_dir, DeploymentMode.AUTOSCALE):
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
        stop_other_mode_services(aws_config, DeploymentMode.AUTOSCALE)
        
        print("\n=== Setting Up Service Discovery ===")
        service_registries = setup_service_discovery(aws_config)
        if service_registries:
            time.sleep(10)
        
        print("\n=== Creating/Updating ECS Services ===")
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
            service_registries=service_registries,
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
        
        print("\n=== Creating Deployment Snapshot ===")
        try:
            import subprocess
            snapshot_cmd = [sys.executable, str(Path(__file__).parent / "snapshot_deployment.py"), "--mode", "autoscale"]
            result = subprocess.run(snapshot_cmd, cwd=services_dir, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("  OK: Deployment snapshot created")
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if 'Snapshot saved' in line or 'Summary saved' in line:
                            print(f"  {line}")
            else:
                print(f"  WARN: Snapshot creation failed: {result.stderr}")
        except Exception as e:
            print(f"  WARN: Could not create snapshot: {e}")
        
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("FAIL: Deployment completed with errors")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()


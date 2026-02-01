#!/usr/bin/env python3
"""
Verify ECS task definition and security group configuration.

Validates EFS configuration in task definitions and security group rules.
"""
import json
import sys
import argparse
import boto3
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import dotenv_values

try:
    import boto3
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)


def load_aws_config() -> Dict:
    """
    Load AWS configuration from aws.env file.
    
    :param: None
    :returns: Dictionary of AWS configuration values.
    """
    current_dir = Path.cwd()
    services_dir = current_dir / "services" if current_dir.name != "services" else current_dir
    
    aws_env_path = services_dir / "aws.env"
    if not aws_env_path.exists():
        print(f"ERROR: aws.env not found at {aws_env_path}")
        print("Please run this script from the services/ directory or ensure aws.env exists.")
        sys.exit(1)
    
    return dict(dotenv_values(str(aws_env_path)))


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_status(status: str, message: str, details: Optional[str] = None):
    """Print a status message with consistent formatting."""
    status_map = {
        "OK": "[OK]",
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
        "INFO": "[INFO]"
    }
    symbol = status_map.get(status, "[?]")
    print(f"  {symbol} {message}")
    if details:
        for line in details.splitlines():
            print(f"    {line}")


def load_task_definition_from_file(file_path: Path) -> Optional[Dict]:
    """
    Load task definition from JSON file.
    
    :param file_path: Path to task definition JSON file.
    :returns: Task definition dictionary or None.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print_status("FAIL", f"Error loading task definition from {file_path}: {e}")
        return None


def get_task_definition_from_aws(ecs_client, task_family: str) -> Optional[Dict]:
    """
    Get latest task definition from AWS ECS.
    
    :param ecs_client: Boto3 ECS client.
    :param task_family: Task definition family name.
    :returns: Task definition dictionary or None.
    """
    try:
        response = ecs_client.describe_task_definition(
            taskDefinition=task_family,
            include=["TAGS"]
        )
        return response.get("taskDefinition")
    except Exception as e:
        print_status("FAIL", f"Error getting task definition from AWS: {e}")
        return None


def check_efs_volumes(task_def: Dict) -> Dict:
    """
    Check EFS volume configuration in task definition.
    
    :param task_def: Task definition dictionary.
    :returns: Dictionary with check results.
    """
    results = {
        "volumes": [],
        "issues": [],
        "warnings": []
    }
    
    volumes = task_def.get("volumes", [])
    efs_volumes = [v for v in volumes if v.get("efsVolumeConfiguration")]
    
    if not efs_volumes:
        results["warnings"].append("No EFS volumes configured in task definition")
        return results
    
    for vol in efs_volumes:
        vol_name = vol.get("name", "UNKNOWN")
        efs_config = vol.get("efsVolumeConfiguration", {})
        file_system_id = efs_config.get("fileSystemId")
        root_directory = efs_config.get("rootDirectory")
        transit_encryption = efs_config.get("transitEncryption", "DISABLED")
        authorization_config = efs_config.get("authorizationConfig", {})
        iam_enabled = authorization_config.get("iam") == "ENABLED"
        
        vol_info = {
            "name": vol_name,
            "fileSystemId": file_system_id,
            "rootDirectory": root_directory,
            "transitEncryption": transit_encryption,
            "iamEnabled": iam_enabled
        }
        results["volumes"].append(vol_info)
        
        if not file_system_id:
            results["issues"].append(f"Volume {vol_name}: Missing fileSystemId")
        
        if root_directory and root_directory != "/":
            results["warnings"].append(
                f"Volume {vol_name}: rootDirectory is '{root_directory}' - "
                f"this directory must exist on the EFS file system"
            )
        
        if transit_encryption == "ENABLED" and not iam_enabled:
            results["warnings"].append(
                f"Volume {vol_name}: Transit encryption enabled but IAM authorization disabled"
            )
    
    return results


def check_mount_points(task_def: Dict) -> Dict:
    """
    Check container mount points configuration.
    
    :param task_def: Task definition dictionary.
    :returns: Dictionary with check results.
    """
    results = {
        "mount_points": [],
        "issues": [],
        "warnings": []
    }
    
    containers = task_def.get("containerDefinitions", [])
    volumes = {v.get("name"): v for v in task_def.get("volumes", [])}
    
    for container in containers:
        container_name = container.get("name", "UNKNOWN")
        mount_points = container.get("mountPoints", [])
        
        for mp in mount_points:
            source_volume = mp.get("sourceVolume")
            container_path = mp.get("containerPath")
            read_only = mp.get("readOnly", False)
            
            vol_info = volumes.get(source_volume)
            is_efs = vol_info and vol_info.get("efsVolumeConfiguration")
            
            mp_info = {
                "container": container_name,
                "sourceVolume": source_volume,
                "containerPath": container_path,
                "readOnly": read_only,
                "isEFS": is_efs
            }
            results["mount_points"].append(mp_info)
            
            if not source_volume:
                results["issues"].append(
                    f"Container {container_name}: Mount point missing sourceVolume"
                )
            
            if not container_path:
                results["issues"].append(
                    f"Container {container_name}: Mount point missing containerPath"
                )
            
            if source_volume and source_volume not in volumes:
                results["issues"].append(
                    f"Container {container_name}: Mount point references non-existent volume '{source_volume}'"
                )
            
            if is_efs and container_path:
                efs_config = vol_info.get("efsVolumeConfiguration", {})
                root_dir = efs_config.get("rootDirectory")
                
                if root_dir and root_dir != "/":
                    results["warnings"].append(
                        f"Container {container_name}: EFS volume '{source_volume}' uses rootDirectory '{root_dir}' - "
                        f"this directory must exist on EFS. Container will see it at '{container_path}'"
                    )
                else:
                    results["warnings"].append(
                        f"Container {container_name}: EFS volume '{source_volume}' mounts EFS root. "
                        f"Container will see it at '{container_path}'. Container should create subdirectories as needed."
                    )
    
    return results


def check_efs_permissions(task_def: Dict, aws_config: Dict) -> Dict:
    """
    Check IAM permissions for EFS access.
    
    :param task_def: Task definition dictionary.
    :param aws_config: AWS configuration dictionary.
    :returns: Dictionary with check results.
    """
    results = {
        "issues": [],
        "warnings": []
    }
    
    execution_role_arn = task_def.get("executionRoleArn", "")
    task_role_arn = task_def.get("taskRoleArn", "")
    
    volumes = task_def.get("volumes", [])
    efs_volumes = [v for v in volumes if v.get("efsVolumeConfiguration")]
    
    if not efs_volumes:
        return results
    
    if not execution_role_arn:
        results["issues"].append("Missing executionRoleArn (required for EFS mounting)")
    else:
        print_status("INFO", f"Execution Role: {execution_role_arn}")
    
    if not task_role_arn:
        results["warnings"].append("Missing taskRoleArn (may be needed for EFS access)")
    else:
        print_status("INFO", f"Task Role: {task_role_arn}")
    
    for vol in efs_volumes:
        efs_config = vol.get("efsVolumeConfiguration", {})
        authorization_config = efs_config.get("authorizationConfig", {})
        iam_enabled = authorization_config.get("iam") == "ENABLED"
        
        if iam_enabled and not execution_role_arn:
            results["issues"].append(
                f"Volume {vol.get('name')}: IAM authorization enabled but no executionRoleArn"
            )
    
    return results


def get_ecs_service_security_groups(ecs_client, cluster: str, service_name: str) -> List[str]:
    """
    Get security groups used by ECS service tasks.
    
    :param ecs_client: Boto3 ECS client.
    :param cluster: ECS cluster name.
    :param service_name: ECS service name.
    :returns: List of security group IDs.
    """
    try:
        response = ecs_client.describe_services(cluster=cluster, services=[service_name])
        services = response.get("services", [])
        if not services:
            return []
        
        service = services[0]
        network_config = service.get("networkConfiguration", {})
        awsvpc_config = network_config.get("awsvpcConfiguration", {})
        return awsvpc_config.get("securityGroups", [])
    except Exception as e:
        print_status("WARN", f"Could not get ECS service security groups: {e}")
        return []


def get_efs_mount_target_security_groups(efs_client, file_system_id: str) -> Dict[str, List[str]]:
    """
    Get security groups for all EFS mount targets.
    
    :param efs_client: Boto3 EFS client.
    :param file_system_id: EFS file system ID.
    :returns: Dictionary mapping mount target ID to list of security group IDs.
    """
    result = {}
    try:
        response = efs_client.describe_mount_targets(FileSystemId=file_system_id)
        mount_targets = response.get("MountTargets", [])
        
        for mt in mount_targets:
            mt_id = mt.get("MountTargetId")
            try:
                sg_response = efs_client.describe_mount_target_security_groups(MountTargetId=mt_id)
                result[mt_id] = sg_response.get("SecurityGroups", [])
            except Exception as e:
                print_status("WARN", f"Could not get security groups for mount target {mt_id}: {e}")
                result[mt_id] = []
    except Exception as e:
        print_status("FAIL", f"Error getting mount target security groups: {e}")
    
    return result


def check_security_group_nfs_rule(ec2_client, sg_id: str, source_sg_ids: List[str]) -> bool:
    """
    Check if security group allows NFS traffic from source security groups.
    
    :param ec2_client: Boto3 EC2 client.
    :param sg_id: Security group ID to check.
    :param source_sg_ids: List of source security group IDs that should have access.
    :returns: True if NFS access is allowed.
    """
    try:
        response = ec2_client.describe_security_groups(GroupIds=[sg_id])
        security_groups = response.get("SecurityGroups", [])
        if not security_groups:
            return False
        
        sg = security_groups[0]
        inbound_rules = sg.get("IpPermissions", [])
        
        allowed_sources = set()
        for rule in inbound_rules:
            if (rule.get("IpProtocol") == "tcp" and 
                rule.get("FromPort") == 2049 and 
                rule.get("ToPort") == 2049):
                for pair in rule.get("UserIdGroupPairs", []):
                    allowed_sources.add(pair.get("GroupId"))
        
        for source_sg_id in source_sg_ids:
            if source_sg_id not in allowed_sources:
                return False
        
        return len(source_sg_ids) > 0 and all(sg_id in allowed_sources for sg_id in source_sg_ids)
    except Exception as e:
        print_status("WARN", f"Error checking security group {sg_id}: {e}")
        return False


def verify_task_definition(aws_config: Dict, task_def: Optional[Dict] = None) -> int:
    """
    Verify task definition EFS configuration.
    
    :param aws_config: AWS configuration dictionary.
    :param task_def: Optional task definition (will load if None).
    :returns: Exit code (0 for success, 1 for failure).
    """
    print("="*70)
    print(" ECS Task Definition EFS Configuration Check")
    print("="*70)
    
    if not task_def:
        region = aws_config.get("AWS_REGION")
        ecs_client = boto3.client("ecs", region_name=region)
        task_family = aws_config.get("ECS_SERVICE_NAME", "euglena").replace("-service", "")
        services_dir = Path.cwd() / "services" if Path.cwd().name != "services" else Path.cwd()
        task_def_file = services_dir / "task-definition-euglena.json"
        
        if task_def_file.exists():
            task_def = load_task_definition_from_file(task_def_file)
        else:
            task_def = get_task_definition_from_aws(ecs_client, task_family)
    
    if not task_def:
        print_status("FAIL", "Could not load task definition")
        sys.exit(1)
    
    family = task_def.get("family", "UNKNOWN")
    revision = task_def.get("revision", "N/A")
    print(f"\nTask Definition: {family}:{revision}")
    
    print_section("EFS Volumes Configuration")
    volume_results = check_efs_volumes(task_def)
    
    if volume_results["volumes"]:
        print(f"  Found {len(volume_results['volumes'])} EFS volume(s):\n")
        for vol in volume_results["volumes"]:
            print(f"  Volume: {vol['name']}")
            print(f"    File System ID: {vol['fileSystemId']}")
            print(f"    Root Directory: {vol['rootDirectory'] or '/'} (root if not specified)")
            print(f"    Transit Encryption: {vol['transitEncryption']}")
            print(f"    IAM Authorization: {'Enabled' if vol['iamEnabled'] else 'Disabled'}")
            print()
    else:
        print_status("INFO", "No EFS volumes configured")
    
    if volume_results["issues"]:
        print_status("FAIL", "EFS Volume Issues:")
        for issue in volume_results["issues"]:
            print(f"    - {issue}")
    
    if volume_results["warnings"]:
        print_status("WARN", "EFS Volume Warnings:")
        for warning in volume_results["warnings"]:
            print(f"    - {warning}")
    
    print_section("Container Mount Points")
    mount_results = check_mount_points(task_def)
    
    if mount_results["mount_points"]:
        efs_mounts = [mp for mp in mount_results["mount_points"] if mp["isEFS"]]
        if efs_mounts:
            print(f"  Found {len(efs_mounts)} EFS mount point(s):\n")
            for mp in efs_mounts:
                print(f"  Container: {mp['container']}")
                print(f"    Source Volume: {mp['sourceVolume']}")
                print(f"    Container Path: {mp['containerPath']}")
                print(f"    Read Only: {mp['readOnly']}")
                print()
        else:
            print_status("INFO", "No EFS mount points found")
    else:
        print_status("INFO", "No mount points configured")
    
    if mount_results["issues"]:
        print_status("FAIL", "Mount Point Issues:")
        for issue in mount_results["issues"]:
            print(f"    - {issue}")
    
    if mount_results["warnings"]:
        print_status("WARN", "Mount Point Warnings:")
        for warning in mount_results["warnings"]:
            print(f"    - {warning}")
    
    print_section("IAM Permissions")
    perm_results = check_efs_permissions(task_def, aws_config)
    
    if perm_results["issues"]:
        print_status("FAIL", "IAM Permission Issues:")
        for issue in perm_results["issues"]:
            print(f"    - {issue}")
    
    if perm_results["warnings"]:
        print_status("WARN", "IAM Permission Warnings:")
        for warning in perm_results["warnings"]:
            print(f"    - {warning}")
    
    print_section("Summary")
    all_issues = volume_results["issues"] + mount_results["issues"] + perm_results["issues"]
    all_warnings = volume_results["warnings"] + mount_results["warnings"] + perm_results["warnings"]
    
    if all_issues:
        print_status("FAIL", f"Found {len(all_issues)} issue(s) that must be fixed")
        return 1
    elif all_warnings:
        print_status("WARN", f"Found {len(all_warnings)} warning(s) - review recommendations")
        return 0
    else:
        print_status("OK", "Task definition EFS configuration looks correct")
        return 0


def verify_security_groups(aws_config: Dict) -> int:
    """
    Verify security group configuration.
    
    :param aws_config: AWS configuration dictionary.
    :returns: Exit code (0 for success, 1 for failure).
    """
    print("="*70)
    print(" Security Group Configuration Verification")
    print("="*70)
    
    region = aws_config.get("AWS_REGION")
    cluster = aws_config.get("ECS_CLUSTER")
    service_name = aws_config.get("ECS_SERVICE_NAME", "euglena-service")
    file_system_id = aws_config.get("EFS_FILE_SYSTEM_ID", "").strip()
    expected_sg_ids = [s.strip() for s in aws_config.get("SECURITY_GROUP_IDS", "").split(",") if s.strip()]
    
    ecs_client = boto3.client("ecs", region_name=region)
    efs_client = boto3.client("efs", region_name=region)
    ec2_client = boto3.client("ec2", region_name=region)
    
    print_section("ECS Service Security Groups")
    ecs_sg_ids = get_ecs_service_security_groups(ecs_client, cluster, service_name)
    
    if ecs_sg_ids:
        print(f"  Service: {service_name}")
        print(f"  Security Groups used by tasks:")
        for sg_id in ecs_sg_ids:
            try:
                sg_response = ec2_client.describe_security_groups(GroupIds=[sg_id])
                sg_name = sg_response.get("SecurityGroups", [{}])[0].get("GroupName", "N/A")
                print(f"    - {sg_id} ({sg_name})")
            except:
                print(f"    - {sg_id}")
        
        if expected_sg_ids:
            missing = set(expected_sg_ids) - set(ecs_sg_ids)
            extra = set(ecs_sg_ids) - set(expected_sg_ids)
            
            if missing:
                print_status("WARN", f"Expected security groups not in service: {', '.join(missing)}")
            if extra:
                print_status("WARN", f"Unexpected security groups in service: {', '.join(extra)}")
            if not missing and not extra:
                print_status("OK", "Service security groups match aws.env configuration")
    else:
        print_status("FAIL", f"Could not get security groups for service {service_name}")
        if expected_sg_ids:
            print(f"  Expected from aws.env: {', '.join(expected_sg_ids)}")
    
    if not file_system_id:
        print_status("INFO", "EFS_FILE_SYSTEM_ID not set, skipping EFS security group checks")
        return 0
    
    print_section("EFS Mount Target Security Groups")
    mt_sg_map = get_efs_mount_target_security_groups(efs_client, file_system_id)
    all_efs_sg_ids = set()
    
    if mt_sg_map:
        for mt_id, sg_ids in mt_sg_map.items():
            all_efs_sg_ids.update(sg_ids)
            print(f"  Mount Target: {mt_id}")
            print(f"    Security Groups: {', '.join(sg_ids) if sg_ids else 'N/A'}")
        
        print_section("NFS Access Verification")
        all_ok = True
        
        for efs_sg_id in all_efs_sg_ids:
            if ecs_sg_ids:
                has_access = check_security_group_nfs_rule(ec2_client, efs_sg_id, ecs_sg_ids)
                if has_access:
                    print_status("OK", f"EFS security group {efs_sg_id} allows NFS (port 2049) from ECS security groups")
                else:
                    print_status("FAIL", f"EFS security group {efs_sg_id} does NOT allow NFS (port 2049) from ECS security groups")
                    print(f"    ECS security groups: {', '.join(ecs_sg_ids)}")
                    print(f"    EFS security group: {efs_sg_id}")
                    all_ok = False
            else:
                print_status("WARN", f"Cannot verify NFS access - no ECS security groups found")
                all_ok = False
        
        if all_ok and ecs_sg_ids:
            print_status("OK", "All EFS mount target security groups allow NFS access from ECS tasks")
        elif not all_ok:
            print_status("FAIL", "Some EFS security groups do not allow NFS access from ECS tasks")
            print("    Action: Add ingress rule to EFS security groups allowing TCP port 2049 from ECS security groups")
    else:
        print_status("WARN", "No mount target security groups found")
    
    print_section("Summary")
    print(f"  ECS Service: {service_name}")
    print(f"  ECS Security Groups: {', '.join(ecs_sg_ids) if ecs_sg_ids else 'N/A'}")
    print(f"  Expected (aws.env): {', '.join(expected_sg_ids) if expected_sg_ids else 'N/A'}")
    print(f"  EFS File System: {file_system_id}")
    print(f"  EFS Mount Target Security Groups: {', '.join(all_efs_sg_ids) if all_efs_sg_ids else 'N/A'}")
    
    if ecs_sg_ids and expected_sg_ids:
        if set(ecs_sg_ids) == set(expected_sg_ids):
            print_status("OK", "Security group configuration verified")
            return 0
        else:
            print_status("FAIL", "Security group mismatch between service and aws.env")
            return 1
    else:
        print_status("WARN", "Could not fully verify security group configuration")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify ECS task definition and security group configuration")
    parser.add_argument("--mode", choices=["task-def", "security-groups", "all"], default="all",
                       help="Verification mode (default: all)")
    parser.add_argument("--task-family", help="Task definition family name (default: from aws.env or 'euglena')")
    parser.add_argument("--file", help="Path to task definition JSON file (overrides AWS lookup)")
    parser.add_argument("--from-aws", action="store_true", help="Get task definition from AWS instead of local file")
    
    args = parser.parse_args()
    
    aws_config = load_aws_config()
    
    exit_code = 0
    
    if args.mode in ["task-def", "all"]:
        task_def = None
        if args.file:
            file_path = Path(args.file)
            if not file_path.is_absolute():
                services_dir = Path.cwd() / "services" if Path.cwd().name != "services" else Path.cwd()
                file_path = services_dir / args.file
            task_def = load_task_definition_from_file(file_path)
        elif args.from_aws:
            region = aws_config.get("AWS_REGION")
            ecs_client = boto3.client("ecs", region_name=region)
            task_family = args.task_family or aws_config.get("ECS_SERVICE_NAME", "euglena").replace("-service", "")
            task_def = get_task_definition_from_aws(ecs_client, task_family)
        
        result = verify_task_definition(aws_config, task_def)
        if result != 0:
            exit_code = result
    
    if args.mode in ["security-groups", "all"]:
        result = verify_security_groups(aws_config)
        if result != 0:
            exit_code = result
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

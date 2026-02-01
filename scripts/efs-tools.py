#!/usr/bin/env python3
"""
EFS verification and diagnostic tools.
"""
import boto3
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dotenv import dotenv_values


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
    
    aws_config = dotenv_values(aws_env_path)
    if not aws_config:
        print(f"ERROR: Failed to load aws.env from {aws_env_path}")
        sys.exit(1)
    
    return aws_config


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


def check_efs_filesystem(efs_client, file_system_id: str) -> Tuple[bool, Dict]:
    """
    Check if EFS filesystem exists and is accessible.
    
    :param efs_client: Boto3 EFS client.
    :param file_system_id: EFS file system ID.
    :returns: Tuple of (success, filesystem dict).
    """
    print_section("EFS File System Check")
    
    try:
        response = efs_client.describe_file_systems(FileSystemId=file_system_id)
        file_systems = response.get("FileSystems", [])
        
        if not file_systems:
            print_status("FAIL", f"EFS filesystem {file_system_id} not found")
            return False, {}
        
        fs = file_systems[0]
        lifecycle_state = fs.get('LifeCycleState') or fs.get('LifecycleState')
        print_status("OK", f"EFS filesystem {file_system_id} found")
        print(f"    Lifecycle State: {lifecycle_state or 'UNKNOWN'}")
        print(f"    Performance Mode: {fs.get('PerformanceMode', 'UNKNOWN')}")
        print(f"    Throughput Mode: {fs.get('ThroughputMode', 'UNKNOWN')}")
        size_info = fs.get('SizeInBytes', {})
        if isinstance(size_info, dict):
            print(f"    Size (bytes): {size_info.get('Value', 'N/A')}")
        else:
            print(f"    Size (bytes): {size_info}")
        
        if lifecycle_state:
            if lifecycle_state == 'available':
                print_status("OK", "EFS filesystem is available")
                return True, fs
            elif lifecycle_state in ('deleting', 'deleted', 'error'):
                print_status("FAIL", f"EFS filesystem is in {lifecycle_state} state")
                return False, fs
            else:
                print_status("WARN", f"EFS filesystem is in {lifecycle_state} state (expected: available)")
                return True, fs
        else:
            print_status("WARN", "EFS filesystem lifecycle state is unknown (filesystem exists)")
            return True, fs
    except Exception as e:
        print_status("FAIL", f"Error checking EFS filesystem: {e}")
        return False, {}


def check_security_group_nfs(ec2_client, security_group_id: str) -> bool:
    """
    Check if security group allows NFS traffic (port 2049).
    
    :param ec2_client: Boto3 EC2 client.
    :param security_group_id: Security group ID.
    :returns: True if NFS allowed.
    """
    print_section("Security Group NFS Access Check")
    
    try:
        response = ec2_client.describe_security_groups(GroupIds=[security_group_id])
        security_groups = response.get("SecurityGroups", [])
        
        if not security_groups:
            print_status("FAIL", f"Security group {security_group_id} not found")
            return False
        
        sg = security_groups[0]
        print_status("OK", f"Security group {security_group_id} found: {sg.get('GroupName', 'N/A')}")
        
        inbound_rules = sg.get("IpPermissions", [])
        nfs_allowed = False
        
        for rule in inbound_rules:
            from_port = rule.get("FromPort")
            to_port = rule.get("ToPort")
            ip_protocol = rule.get("IpProtocol")
            
            if ip_protocol == "tcp" and from_port and to_port:
                if from_port <= 2049 <= to_port:
                    nfs_allowed = True
                    print_status("OK", f"NFS traffic (TCP 2049) allowed")
                    print(f"    Port Range: {from_port}-{to_port}")
                    print(f"    Source: {rule.get('UserIdGroupPairs', [{}])[0].get('GroupId', 'N/A')}")
                    break
        
        if not nfs_allowed:
            print_status("FAIL", "NFS traffic (TCP port 2049) not allowed in security group")
            print("    EFS requires TCP port 2049 to be open for NFS traffic")
            return False
        
        return True
    except Exception as e:
        print_status("FAIL", f"Error checking security group: {e}")
        return False


def check_mount_targets(efs_client, file_system_id: str, subnet_ids: List[str]) -> bool:
    """
    Check if mount targets exist in all required subnets and are in available state.
    
    :param efs_client: Boto3 EFS client.
    :param file_system_id: EFS file system ID.
    :param subnet_ids: List of subnet IDs where ECS tasks run.
    :returns: True if all mount targets exist and are available.
    """
    print_section("EFS Mount Targets Check")
    
    try:
        response = efs_client.describe_mount_targets(FileSystemId=file_system_id)
        mount_targets = response.get("MountTargets", [])
        
        print(f"  Found {len(mount_targets)} mount target(s)")
        
        if not mount_targets:
            print_status("FAIL", "No mount targets found for EFS filesystem")
            print("    Mount targets must exist in each subnet used by ECS tasks")
            return False
        
        mount_target_subnets = {mt.get("SubnetId") for mt in mount_targets}
        required_subnets = set(subnet_ids)
        missing_subnets = required_subnets - mount_target_subnets
        
        all_available = True
        for mt in mount_targets:
            subnet_id = mt.get("SubnetId")
            mount_target_id = mt.get("MountTargetId")
            ip_address = mt.get("IpAddress", "N/A")
            
            state = mt.get("LifeCycleState") or mt.get("LifecycleState")
            if not state:
                try:
                    mt_detail_response = efs_client.describe_mount_targets(MountTargetId=mount_target_id)
                    mt_details = mt_detail_response.get("MountTargets", [])
                    if mt_details:
                        state = mt_details[0].get("LifeCycleState") or mt_details[0].get("LifecycleState")
                except Exception as e:
                    state = None
            
            if not state:
                state = "UNKNOWN"
            
            state_lower = state.lower() if state else "unknown"
            
            if subnet_id in required_subnets:
                if state_lower == "available":
                    print_status("OK", f"Mount target {mount_target_id} in subnet {subnet_id} is available (IP: {ip_address})")
                elif state and state_lower != "unknown":
                    print_status("FAIL", f"Mount target {mount_target_id} in subnet {subnet_id} is in {state} state (must be 'available')")
                    all_available = False
                else:
                    print_status("FAIL", f"Mount target {mount_target_id} in subnet {subnet_id} lifecycle state is unknown (raw value: {repr(state)})")
                    all_available = False
            else:
                print_status("WARN", f"Mount target {mount_target_id} in subnet {subnet_id} (not in required subnets)")
        
        if missing_subnets:
            print_status("FAIL", f"Missing mount targets for subnets: {', '.join(missing_subnets)}")
            print("    Each subnet used by ECS tasks must have a mount target")
            return False
        
        if not all_available:
            print_status("FAIL", "Some mount targets are not in 'available' state")
            print("    All mount targets must be in 'available' state for ECS tasks to mount EFS")
            return False
        
        for mt in mount_targets:
            if mt.get("SubnetId") in required_subnets:
                mt_id = mt.get("MountTargetId")
                state = mt.get("LifeCycleState") or mt.get("LifecycleState") or ""
                if state.lower() != "available":
                    continue
                try:
                    mt_response = efs_client.describe_mount_target_security_groups(MountTargetId=mt_id)
                    mt_sgs = mt_response.get("SecurityGroups", [])
                    print(f"    Mount target {mt_id} security groups: {', '.join(mt_sgs)}")
                except Exception as e:
                    print_status("WARN", f"Could not get security groups for mount target {mt_id}: {e}")
        
        return all_available
    except Exception as e:
        print_status("FAIL", f"Error checking mount targets: {e}")
        return False


def check_subnet_routing(ec2_client, subnet_ids: List[str], vpc_id: str) -> bool:
    """
    Check if subnets have routes to EFS (via NAT Gateway or Internet Gateway).
    
    :param ec2_client: Boto3 EC2 client.
    :param subnet_ids: List of subnet IDs.
    :param vpc_id: VPC ID.
    :returns: True if routing is configured.
    """
    print_section("Subnet Routing Check")
    
    try:
        response = ec2_client.describe_route_tables(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]}
            ]
        )
        route_tables = response.get("RouteTables", [])
        
        all_ok = True
        for subnet_id in subnet_ids:
            print(f"\n  Checking subnet {subnet_id}:")
            
            route_table = None
            for rt in route_tables:
                associations = rt.get("Associations", [])
                for assoc in associations:
                    if assoc.get("SubnetId") == subnet_id:
                        route_table = rt
                        break
                if route_table:
                    break
            
            if not route_table:
                for rt in route_tables:
                    associations = rt.get("Associations", [])
                    if any(a.get("Main", False) for a in associations):
                        route_table = rt
                        break
            
            if not route_table:
                print_status("WARN", f"No route table found for subnet {subnet_id}")
                all_ok = False
                continue
            
            routes = route_table.get("Routes", [])
            has_nat = False
            has_igw = False
            
            for route in routes:
                gateway_id = route.get("GatewayId", "")
                nat_gateway_id = route.get("NatGatewayId", "")
                destination = route.get("DestinationCidrBlock", "")
                
                if destination == "0.0.0.0/0":
                    if gateway_id.startswith("igw-"):
                        has_igw = True
                        print_status("OK", f"Subnet has route to Internet Gateway {gateway_id}")
                    elif nat_gateway_id:
                        has_nat = True
                        print_status("OK", f"Subnet has route to NAT Gateway {nat_gateway_id}")
            
            if not has_nat and not has_igw:
                print_status("WARN", f"Subnet {subnet_id} may not have internet access for EFS utils")
                print("    EFS utils need to download from internet during container initialization")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print_status("FAIL", f"Error checking subnet routing: {e}")
        return False


def check_task_execution_role_permissions(iam_client, role_name: str, file_system_id: str) -> bool:
    """
    Check if task execution role has EFS permissions.
    
    :param iam_client: Boto3 IAM client.
    :param role_name: IAM role name.
    :param file_system_id: EFS file system ID.
    :returns: True if permissions exist.
    """
    print_section("Task Execution Role Permissions Check")
    
    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response.get("Role", {}).get("Arn")
        
        if not role_arn:
            print_status("FAIL", f"Task execution role {role_name} not found")
            return False
        
        print_status("OK", f"Task execution role {role_name} found: {role_arn}")
        
        policies_response = iam_client.list_attached_role_policies(RoleName=role_name)
        attached_policies = policies_response.get("AttachedPolicies", [])
        
        has_efs_policy = False
        for policy in attached_policies:
            policy_arn = policy.get("PolicyArn", "")
            if "EFS" in policy_arn or "efs" in policy_arn.lower():
                has_efs_policy = True
                print_status("OK", f"EFS policy attached: {policy_arn}")
                break
        
        if not has_efs_policy:
            inline_policies_response = iam_client.list_role_policies(RoleName=role_name)
            inline_policies = inline_policies_response.get("PolicyNames", [])
            
            for policy_name in inline_policies:
                policy_response = iam_client.get_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name
                )
                policy_doc = policy_response.get("PolicyDocument", {})
                policy_str = str(policy_doc)
                
                if "elasticfilesystem" in policy_str.lower() or "efs" in policy_str.lower():
                    has_efs_policy = True
                    print_status("OK", f"EFS permissions found in inline policy: {policy_name}")
                    break
        
        if not has_efs_policy:
            print_status("WARN", "No EFS-specific policy found")
            print("    Task execution role should have permissions for:")
            print("      - elasticfilesystem:ClientMount")
            print("      - elasticfilesystem:ClientWrite")
            print("      - elasticfilesystem:ClientRootAccess")
        
        has_execution_policy = any(
            "AmazonECSTaskExecutionRolePolicy" in p.get("PolicyArn", "")
            for p in attached_policies
        )
        
        if has_execution_policy:
            print_status("OK", "AmazonECSTaskExecutionRolePolicy attached (includes basic EFS permissions)")
        else:
            print_status("WARN", "AmazonECSTaskExecutionRolePolicy not attached")
        
        return True
    except Exception as e:
        print_status("FAIL", f"Error checking task execution role: {e}")
        return False


def get_mount_target_details(efs_client, file_system_id: str) -> List[Dict]:
    """
    Get detailed information about all mount targets.
    
    :param efs_client: Boto3 EFS client.
    :param file_system_id: EFS file system ID.
    :returns: List of mount target dictionaries.
    """
    mount_targets = []
    try:
        response = efs_client.describe_mount_targets(FileSystemId=file_system_id)
        mt_list = response.get("MountTargets", [])
        
        for mt in mt_list:
            mt_id = mt.get("MountTargetId")
            state = mt.get("LifeCycleState") or mt.get("LifecycleState")
            
            if not state or state == "UNKNOWN":
                try:
                    mt_detail_response = efs_client.describe_mount_targets(MountTargetId=mt_id)
                    mt_details = mt_detail_response.get("MountTargets", [])
                    if mt_details:
                        full_mt = mt_details[0]
                        retrieved_state = full_mt.get("LifeCycleState") or full_mt.get("LifecycleState")
                        if retrieved_state:
                            state = retrieved_state
                            mt["LifeCycleState"] = retrieved_state
                            mt["LifecycleState"] = retrieved_state
                        mt.update(full_mt)
                except Exception as e:
                    print_status("WARN", f"Could not get details for mount target {mt_id}: {e}")
            
            if not state:
                state = "UNKNOWN"
                mt["LifeCycleState"] = state
                mt["LifecycleState"] = state
            
            try:
                sg_response = efs_client.describe_mount_target_security_groups(MountTargetId=mt_id)
                mt["SecurityGroups"] = sg_response.get("SecurityGroups", [])
            except:
                mt["SecurityGroups"] = []
            
            mount_targets.append(mt)
    except Exception as e:
        print_status("FAIL", f"Error getting mount targets: {e}")
    
    return mount_targets


def get_access_points(efs_client, file_system_id: str) -> List[Dict]:
    """
    Get EFS access points for the file system.
    
    :param efs_client: Boto3 EFS client.
    :param file_system_id: EFS file system ID.
    :returns: List of access point dictionaries.
    """
    try:
        response = efs_client.describe_access_points(FileSystemId=file_system_id)
        return response.get("AccessPoints", [])
    except Exception as e:
        print_status("WARN", f"Could not get access points: {e}")
        return []


def get_efs_policy(efs_client, file_system_id: str) -> Optional[Dict]:
    """
    Get EFS file system policy.
    
    :param efs_client: Boto3 EFS client.
    :param file_system_id: EFS file system ID.
    :returns: Policy document dictionary or None.
    """
    try:
        response = efs_client.describe_file_system_policy(FileSystemId=file_system_id)
        policy_str = response.get("Policy", "{}")
        return json.loads(policy_str)
    except efs_client.exceptions.PolicyNotFoundException:
        return None
    except Exception as e:
        print_status("WARN", f"Could not get EFS policy: {e}")
        return None


def get_ecs_tasks_with_efs_errors(ecs_client, cluster: str, service_name: str, hours: int = 24) -> List[Dict]:
    """
    Get ECS tasks that failed due to EFS-related errors.
    
    :param ecs_client: Boto3 ECS client.
    :param cluster: ECS cluster name.
    :param service_name: ECS service name.
    :param hours: Number of hours to look back.
    :returns: List of task dictionaries with EFS errors.
    """
    failed_tasks = []
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        response = ecs_client.list_tasks(cluster=cluster, serviceName=service_name, desiredStatus="STOPPED")
        task_arns = response.get("taskArns", [])
        
        if not task_arns:
            return []
        
        tasks_response = ecs_client.describe_tasks(cluster=cluster, tasks=task_arns[:100])
        tasks = tasks_response.get("tasks", [])
        
        for task in tasks:
            stopped_reason = task.get("stoppedReason", "")
            if "EFS" in stopped_reason.upper() or "mount" in stopped_reason.lower():
                stopped_at = task.get("stoppedAt")
                if stopped_at:
                    stopped_dt = datetime.fromisoformat(stopped_at.replace('Z', '+00:00'))
                    if stopped_dt.replace(tzinfo=None) >= cutoff_time:
                        failed_tasks.append(task)
    except Exception as e:
        print_status("WARN", f"Could not get ECS tasks with EFS errors: {e}")
    
    return failed_tasks


def format_size(bytes_value: int) -> str:
    """
    Format bytes to human-readable size.
    
    :param bytes_value: Size in bytes.
    :returns: Formatted size string.
    """
    if bytes_value >= 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"
    elif bytes_value >= 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.2f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f} KB"
    else:
        return f"{bytes_value} B"


def verify_efs(aws_config: Dict) -> int:
    """
    Verify EFS configuration.
    
    :param aws_config: AWS configuration dictionary.
    :returns: Exit code (0 for success, 1 for failure).
    """
    print("="*70)
    print(" EFS Connectivity and Configuration Verification")
    print("="*70)
    
    region = aws_config.get("AWS_REGION")
    file_system_id = aws_config.get("EFS_FILE_SYSTEM_ID", "").strip()
    vpc_id = aws_config.get("VPC_ID", "").strip()
    subnet_ids_str = aws_config.get("SUBNET_IDS", "").strip()
    security_group_ids_str = aws_config.get("SECURITY_GROUP_IDS", "").strip()
    task_execution_role = aws_config.get("ECS_TASK_EXECUTION_ROLE_NAME", "ecsTaskExecutionRole")
    
    if not file_system_id:
        print("ERROR: EFS_FILE_SYSTEM_ID not set in aws.env")
        sys.exit(1)
    
    if not vpc_id:
        print("ERROR: VPC_ID not set in aws.env")
        sys.exit(1)
    
    if not subnet_ids_str:
        print("ERROR: SUBNET_IDS not set in aws.env")
        sys.exit(1)
    
    subnet_ids = [s.strip() for s in subnet_ids_str.split(",") if s.strip()]
    security_group_ids = [s.strip() for s in security_group_ids_str.split(",") if s.strip()]
    
    if not subnet_ids:
        print("ERROR: No valid subnet IDs found in SUBNET_IDS")
        sys.exit(1)
    
    if not security_group_ids:
        print("ERROR: No valid security group IDs found in SECURITY_GROUP_IDS")
        sys.exit(1)
    
    efs_client = boto3.client("efs", region_name=region)
    ec2_client = boto3.client("ec2", region_name=region)
    iam_client = boto3.client("iam", region_name=region)
    
    results = []
    
    fs_ok, fs_info = check_efs_filesystem(efs_client, file_system_id)
    results.append(("EFS Filesystem", fs_ok))
    
    mt_ok = True
    if fs_ok:
        mt_ok = check_mount_targets(efs_client, file_system_id, subnet_ids)
        results.append(("Mount Targets", mt_ok))
    else:
        results.append(("Mount Targets", False))
    
    for sg_id in security_group_ids:
        sg_ok = check_security_group_nfs(ec2_client, sg_id)
        results.append((f"Security Group {sg_id}", sg_ok))
    
    routing_ok = check_subnet_routing(ec2_client, subnet_ids, vpc_id)
    results.append(("Subnet Routing", routing_ok))
    
    role_ok = check_task_execution_role_permissions(iam_client, task_execution_role, file_system_id)
    results.append(("Task Execution Role", role_ok))
    
    print_section("Verification Summary")
    all_passed = True
    for check_name, passed in results:
        status = "OK" if passed else "FAIL"
        print_status(status, check_name)
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[OK] All EFS checks passed! EFS should be accessible from ECS tasks.")
        return 0
    else:
        print("\n[FAIL] Some EFS checks failed. Please review the issues above.")
        return 1


def diagnose_efs(aws_config: Dict, file_system_id: Optional[str] = None, hours: int = 24) -> None:
    """
    Diagnose EFS issues with detailed information.
    
    :param aws_config: AWS configuration dictionary.
    :param file_system_id: Optional EFS file system ID override.
    :param hours: Hours to look back for ECS task errors.
    """
    print("="*70)
    print(" EFS Comprehensive Diagnostic")
    print("="*70)
    
    region = aws_config.get("AWS_REGION")
    fs_id = file_system_id or aws_config.get("EFS_FILE_SYSTEM_ID", "").strip()
    
    if not fs_id:
        print("ERROR: EFS_FILE_SYSTEM_ID not set in aws.env and --file-system-id not provided")
        sys.exit(1)
    
    efs_client = boto3.client("efs", region_name=region)
    ecs_client = boto3.client("ecs", region_name=region)
    
    print_section("EFS File System Details")
    response = efs_client.describe_file_systems(FileSystemId=fs_id)
    file_systems = response.get("FileSystems", [])
    if file_systems:
        fs = file_systems[0]
        print(f"  File System ID: {fs.get('FileSystemId')}")
        lifecycle_state = fs.get('LifeCycleState') or fs.get('LifecycleState') or 'UNKNOWN'
        print(f"  Lifecycle State: {lifecycle_state}")
        print(f"  Performance Mode: {fs.get('PerformanceMode', 'UNKNOWN')}")
        print(f"  Throughput Mode: {fs.get('ThroughputMode', 'UNKNOWN')}")
        print(f"  Encrypted: {fs.get('Encrypted', False)}")
        
        size_info = fs.get('SizeInBytes', {})
        if isinstance(size_info, dict):
            size_value = size_info.get('Value', 0)
            size_timestamp = size_info.get('Timestamp', 'N/A')
            print(f"  Size: {format_size(size_value)} (as of {size_timestamp})")
        
        print(f"  Creation Time: {fs.get('CreationTime', 'N/A')}")
        print(f"  DNS Name: {fs.get('FileSystemId')}.efs.{region}.amazonaws.com")
        
        tags = fs.get('Tags', [])
        if tags:
            print(f"  Tags:")
            for tag in tags:
                print(f"    {tag.get('Key')}: {tag.get('Value')}")
    else:
        print_status("FAIL", f"Could not retrieve details for file system {fs_id}")
        sys.exit(1)
    
    print_section("Mount Targets")
    mount_targets = get_mount_target_details(efs_client, fs_id)
    if mount_targets:
        print(f"  Found {len(mount_targets)} mount target(s):\n")
        for mt in mount_targets:
            mt_id = mt.get("MountTargetId")
            subnet_id = mt.get("SubnetId")
            state = mt.get("LifeCycleState") or mt.get("LifecycleState") or "UNKNOWN"
            ip_address = mt.get("IpAddress", "N/A")
            availability_zone = mt.get("AvailabilityZoneId", "N/A")
            network_interface_id = mt.get("NetworkInterfaceId", "N/A")
            security_groups = mt.get("SecurityGroups", [])
            
            status = "OK" if state.lower() == "available" else "FAIL"
            print_status(status, f"Mount Target: {mt_id}")
            print(f"    Subnet: {subnet_id}")
            print(f"    Availability Zone: {availability_zone}")
            print(f"    State: {state}")
            print(f"    IP Address: {ip_address}")
            print(f"    Network Interface: {network_interface_id}")
            print(f"    Security Groups: {', '.join(security_groups) if security_groups else 'N/A'}")
            print()
    else:
        print_status("FAIL", "No mount targets found")
    
    print_section("Access Points")
    access_points = get_access_points(efs_client, fs_id)
    if access_points:
        print(f"  Found {len(access_points)} access point(s):\n")
        for ap in access_points:
            ap_id = ap.get("AccessPointId")
            ap_arn = ap.get("AccessPointArn", "N/A")
            root_dir = ap.get("RootDirectory", {}).get("Path", "/")
            print(f"  Access Point: {ap_id}")
            print(f"    ARN: {ap_arn}")
            print(f"    Root Directory: {root_dir}")
            print()
    else:
        print_status("INFO", "No access points configured (using root directory)")
    
    print_section("EFS File System Policy")
    policy = get_efs_policy(efs_client, fs_id)
    if policy:
        print("  Policy Document:")
        print(json.dumps(policy, indent=4))
    else:
        print_status("INFO", "No file system policy configured")
    
    print_section("Recent ECS Task Failures (EFS-related)")
    cluster = aws_config.get("ECS_CLUSTER")
    service_name = aws_config.get("ECS_SERVICE_NAME", "euglena-service")
    
    failed_tasks = get_ecs_tasks_with_efs_errors(ecs_client, cluster, service_name, hours)
    if failed_tasks:
        print(f"  Found {len(failed_tasks)} task(s) with EFS-related failures in last {hours} hours:\n")
        for task in failed_tasks[:10]:
            task_id = task.get("taskArn", "").split("/")[-1]
            stopped_reason = task.get("stoppedReason", "N/A")
            stopped_at = task.get("stoppedAt", "N/A")
            print(f"  Task: {task_id}")
            print(f"    Stopped At: {stopped_at}")
            print(f"    Reason: {stopped_reason}")
            print()
    else:
        print_status("OK", f"No EFS-related task failures in last {hours} hours")
    
    print_section("Summary")
    print(f"  File System: {fs_id}")
    print(f"  Mount Targets: {len(mount_targets)}")
    available_mts = [mt for mt in mount_targets if (mt.get("LifeCycleState") or mt.get("LifecycleState") or "").lower() == "available"]
    print(f"  Available Mount Targets: {len(available_mts)}")
    print(f"  Access Points: {len(access_points)}")
    print(f"  Recent EFS Task Failures: {len(failed_tasks)}")
    
    if len(available_mts) == len(mount_targets) and len(mount_targets) > 0:
        print_status("OK", "All mount targets are available")
    else:
        print_status("FAIL", f"Only {len(available_mts)}/{len(mount_targets)} mount targets are available")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EFS verification and diagnostic tools")
    parser.add_argument("--mode", choices=["verify", "diagnose"], default="verify",
                       help="Mode: verify (default) or diagnose")
    parser.add_argument("--file-system-id", help="EFS file system ID (overrides aws.env)")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back for ECS task errors (diagnose mode)")
    
    args = parser.parse_args()
    
    aws_config = load_aws_config()
    
    if args.mode == "verify":
        sys.exit(verify_efs(aws_config))
    else:
        diagnose_efs(aws_config, args.file_system_id, args.hours)


if __name__ == "__main__":
    main()

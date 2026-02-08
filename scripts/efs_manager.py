"""
EFS (Elastic File System) management utilities.

Handles EFS file system discovery and mount target creation/verification.
Can be run directly or imported as a module.
"""
import boto3
from typing import Dict, List, Optional
import sys


class EfsManager:
    """
    Manages EFS file systems and mount targets.
    
    :param region: AWS region name.
    :param file_system_id: Optional EFS file system ID.
    """
    
    def __init__(self, region: str, file_system_id: Optional[str] = None):
        self.region = region
        self.file_system_id = file_system_id
        self.efs_client = boto3.client("efs", region_name=region)
        self.ec2_client = boto3.client("ec2", region_name=region)
    
    def get_file_system(self, file_system_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get EFS file system information.
        
        :param file_system_id: EFS file system ID (uses self.file_system_id if None).
        :returns: File system dictionary or None if not found.
        """
        fs_id = file_system_id or self.file_system_id
        if not fs_id:
            return None
        
        try:
            response = self.efs_client.describe_file_systems(FileSystemId=fs_id)
            file_systems = response.get("FileSystems", [])
            if file_systems:
                return file_systems[0]
        except Exception as e:
            print(f"Error getting EFS file system {fs_id}: {e}", file=sys.stderr)
        return None
    
    def get_mount_targets(self, file_system_id: Optional[str] = None) -> List[Dict]:
        """
        Get all mount targets for an EFS file system.
        
        :param file_system_id: EFS file system ID (uses self.file_system_id if None).
        :returns: List of mount target dictionaries.
        """
        fs_id = file_system_id or self.file_system_id
        if not fs_id:
            return []
        
        try:
            response = self.efs_client.describe_mount_targets(FileSystemId=fs_id)
            return response.get("MountTargets", [])
        except Exception as e:
            print(f"Error getting mount targets for {fs_id}: {e}", file=sys.stderr)
            return []
    
    def find_mount_target_for_subnet(self, file_system_id: Optional[str] = None, subnet_id: str = None) -> Optional[Dict]:
        """
        Find mount target for a specific subnet.
        
        :param file_system_id: EFS file system ID (uses self.file_system_id if None).
        :param subnet_id: Subnet ID to search for.
        :returns: Mount target dictionary or None if not found.
        """
        if not subnet_id:
            return None
        
        mount_targets = self.get_mount_targets(file_system_id)
        for mt in mount_targets:
            if mt.get("SubnetId") == subnet_id:
                return mt
        return None
    
    def create_mount_target(
        self,
        file_system_id: Optional[str] = None,
        subnet_id: str = None,
        security_group_ids: List[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create a mount target for an EFS file system in a subnet.
        
        :param file_system_id: EFS file system ID (uses self.file_system_id if None).
        :param subnet_id: Subnet ID where mount target will be created.
        :param security_group_ids: List of security group IDs for the mount target.
        :param ip_address: Optional IP address for the mount target.
        :returns: Created mount target dictionary or None on failure.
        """
        fs_id = file_system_id or self.file_system_id
        if not fs_id or not subnet_id:
            return None
        
        existing = self.find_mount_target_for_subnet(fs_id, subnet_id)
        if existing:
            mount_target_id = existing.get("MountTargetId") if isinstance(existing, dict) else existing
            print(f"Mount target already exists in subnet {subnet_id}: {mount_target_id}")
            return existing
        
        try:
            params = {
                "FileSystemId": fs_id,
                "SubnetId": subnet_id
            }
            
            if security_group_ids:
                params["SecurityGroups"] = security_group_ids
            
            if ip_address:
                params["IpAddress"] = ip_address
            
            response = self.efs_client.create_mount_target(**params)
            mount_target = response.get("MountTargetId")
            print(f"Created mount target {mount_target} in subnet {subnet_id}")
            return response
        except Exception as e:
            print(f"Error creating mount target in subnet {subnet_id}: {e}", file=sys.stderr)
            return None
    
    def ensure_mount_targets(
        self,
        file_system_id: Optional[str] = None,
        subnet_ids: List[str] = None,
        security_group_ids: List[str] = None
    ) -> List[Dict]:
        """
        Ensure mount targets exist for all specified subnets.
        
        :param file_system_id: EFS file system ID (uses self.file_system_id if None).
        :param subnet_ids: List of subnet IDs where mount targets should exist.
        :param security_group_ids: List of security group IDs for mount targets.
        :returns: List of mount target dictionaries (existing or newly created).
        """
        fs_id = file_system_id or self.file_system_id
        if not fs_id or not subnet_ids:
            return []
        
        mount_targets = []
        for subnet_id in subnet_ids:
            mt = self.find_mount_target_for_subnet(fs_id, subnet_id)
            if not mt:
                mt = self.create_mount_target(fs_id, subnet_id, security_group_ids)
            if mt:
                mount_target_id = mt.get("MountTargetId") if isinstance(mt, dict) else mt
                mount_targets.append(mt)
        
        return mount_targets
    
    def get_mount_target_security_groups(self, file_system_id: Optional[str] = None) -> List[str]:
        """
        Get all unique security group IDs from existing mount targets.
        
        :param file_system_id: EFS file system ID (uses self.file_system_id if None).
        :returns: List of unique security group IDs.
        """
        fs_id = file_system_id or self.file_system_id
        if not fs_id:
            return []
        
        mount_targets = self.get_mount_targets(fs_id)
        security_group_ids = set()
        
        for mt in mount_targets:
            mt_id = mt.get("MountTargetId")
            if mt_id:
                try:
                    response = self.efs_client.describe_mount_target_security_groups(MountTargetId=mt_id)
                    sg_ids = response.get("SecurityGroups", [])
                    security_group_ids.update(sg_ids)
                except Exception as e:
                    print(f"  WARN: Could not get security groups for mount target {mt_id}: {e}", file=sys.stderr)
        
        return list(security_group_ids)
    
    def get_or_create_security_group_for_efs(self, vpc_id: str, ecs_security_group_ids: List[str] = None, security_group_name: str = "efs-mount-target-sg") -> Optional[str]:
        """
        Get or create a security group for EFS mount targets.
        
        :param vpc_id: VPC ID where security group should be created.
        :param ecs_security_group_ids: List of ECS task security group IDs that need access to EFS.
        :param security_group_name: Name for the security group.
        :returns: Security group ID or None on failure.
        """
        try:
            response = self.ec2_client.describe_security_groups(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    {"Name": "group-name", "Values": [security_group_name]}
                ]
            )
            sgs = response.get("SecurityGroups", [])
            if sgs:
                sg_id = sgs[0]["GroupId"]
                print(f"Using existing security group {sg_id} for EFS mount targets")
                
                if ecs_security_group_ids:
                    self._ensure_ecs_access_to_efs(sg_id, ecs_security_group_ids)
                
                return sg_id
            
            response = self.ec2_client.create_security_group(
                GroupName=security_group_name,
                Description="Security group for EFS mount targets",
                VpcId=vpc_id
            )
            sg_id = response["GroupId"]
            
            user_group_pairs = []
            if ecs_security_group_ids:
                for ecs_sg_id in ecs_security_group_ids:
                    user_group_pairs.append({"GroupId": ecs_sg_id})
            else:
                user_group_pairs.append({"GroupId": sg_id})
            
            self.ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 2049,
                        "ToPort": 2049,
                        "UserIdGroupPairs": user_group_pairs
                    }
                ]
            )
            
            print(f"Created security group {sg_id} for EFS mount targets")
            return sg_id
        except Exception as e:
            print(f"Error creating security group for EFS: {e}", file=sys.stderr)
            return None
    
    def _ensure_ecs_access_to_efs(self, efs_sg_id: str, ecs_security_group_ids: List[str]):
        """
        Ensure ECS security groups have access to EFS mount target.
        
        Checks existing rules and only adds missing ones. Handles duplicate rules gracefully.
        
        :param efs_sg_id: EFS mount target security group ID.
        :param ecs_security_group_ids: List of ECS task security group IDs.
        """
        try:
            response = self.ec2_client.describe_security_groups(GroupIds=[efs_sg_id])
            current_sg = response.get("SecurityGroups", [{}])[0]
            current_rules = current_sg.get("IpPermissions", [])
            
            allowed_sg_ids = set()
            for rule in current_rules:
                if (rule.get("IpProtocol") == "tcp" and 
                    rule.get("FromPort") == 2049 and 
                    rule.get("ToPort") == 2049):
                    for pair in rule.get("UserIdGroupPairs", []):
                        allowed_sg_ids.add(pair.get("GroupId"))
            
            missing_sg_ids = [sg_id for sg_id in ecs_security_group_ids if sg_id not in allowed_sg_ids]
            if missing_sg_ids:
                user_group_pairs = [{"GroupId": sg_id} for sg_id in missing_sg_ids]
                try:
                    self.ec2_client.authorize_security_group_ingress(
                        GroupId=efs_sg_id,
                        IpPermissions=[
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 2049,
                                "ToPort": 2049,
                                "UserIdGroupPairs": user_group_pairs
                            }
                        ]
                    )
                    print(f"  Added NFS access from ECS security groups: {', '.join(missing_sg_ids)}")
                except Exception as add_error:
                    error_str = str(add_error)
                    if "already exists" in error_str or "Duplicate" in error_str or "InvalidPermission.Duplicate" in error_str:
                        print(f"  OK: NFS access rules already exist (no changes needed)")
                    else:
                        raise
            else:
                print(f"  OK: NFS access already configured for all ECS security groups")
        except Exception as e:
            error_str = str(e)
            if "already exists" in error_str or "Duplicate" in error_str:
                print(f"  OK: Security group rules already exist (no changes needed)")
            else:
                print(f"  WARN: Could not update EFS security group rules: {e}", file=sys.stderr)


def parse_args():
    """
    Parse CLI arguments.

    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Manage EFS mount targets")
    parser.add_argument("--file-system-id", help="EFS file system ID")
    parser.add_argument("--subnet-ids", help="Comma-separated list of subnet IDs")
    parser.add_argument("--security-group-ids", help="Comma-separated list of security group IDs for EFS mount targets")
    parser.add_argument("--ecs-security-group-ids", help="Comma-separated list of ECS task security group IDs that need EFS access")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env (defaults to current directory)")
    
    return parser.parse_args()

def main():
    """
    Main entry point when run directly.
    Sets up EFS mount targets based on aws.env configuration.
    """
    import argparse
    from pathlib import Path
    from dotenv import dotenv_values
    
    args = parse_args()
    
    services_dir = args.services_dir or Path.cwd()
    aws_env_path = services_dir / "aws.env"
    
    if aws_env_path.exists():
        aws_config = dict(dotenv_values(str(aws_env_path)))
        region = aws_config.get("AWS_REGION")
        file_system_id = args.file_system_id or aws_config.get("EFS_FILE_SYSTEM_ID", "").strip()
        subnet_ids_str = args.subnet_ids or aws_config.get("SUBNET_IDS", "")
        security_group_ids_str = args.security_group_ids or aws_config.get("SECURITY_GROUP_IDS", "")
        vpc_id = aws_config.get("VPC_ID")
    else:
        if not args.file_system_id or not args.subnet_ids:
            print(f"Error: --file-system-id and --subnet-ids required or aws.env must exist", file=sys.stderr)
            sys.exit(1)
        region = "us-east-1"
        file_system_id = args.file_system_id
        subnet_ids_str = args.subnet_ids
        security_group_ids_str = args.security_group_ids
        vpc_id = None
    
    if not file_system_id:
        print("Error: EFS file system ID not found. Set EFS_FILE_SYSTEM_ID in aws.env", file=sys.stderr)
        sys.exit(1)
    
    subnet_ids = [s.strip() for s in subnet_ids_str.split(",") if s.strip()] if subnet_ids_str else []
    security_group_ids = [s.strip() for s in security_group_ids_str.split(",") if s.strip()] if security_group_ids_str else []
    ecs_security_group_ids_str = args.ecs_security_group_ids or aws_config.get("SECURITY_GROUP_IDS", "") if aws_env_path.exists() else ""
    ecs_security_group_ids = [s.strip() for s in ecs_security_group_ids_str.split(",") if s.strip()] if ecs_security_group_ids_str else []
    
    if not subnet_ids:
        print("Error: No subnet IDs provided", file=sys.stderr)
        sys.exit(1)
    
    manager = EfsManager(region=region, file_system_id=file_system_id)
    
    fs = manager.get_file_system()
    if not fs:
        print(f"Error: EFS file system {file_system_id} not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"=== EFS File System: {file_system_id} ===")
    print(f"State: {fs.get('LifeCycleState', 'N/A')}")
    print(f"Performance Mode: {fs.get('PerformanceMode', 'N/A')}")
    
    if not security_group_ids and vpc_id:
        sg_id = manager.get_or_create_security_group_for_efs(vpc_id, ecs_security_group_ids=ecs_security_group_ids)
        if sg_id:
            security_group_ids = [sg_id]
    elif security_group_ids and vpc_id and ecs_security_group_ids:
        efs_sg_id = security_group_ids[0] if security_group_ids else None
        if efs_sg_id:
            manager._ensure_ecs_access_to_efs(efs_sg_id, ecs_security_group_ids)
    
    print(f"\n=== Ensuring Mount Targets ===")
    print(f"Subnets: {', '.join(subnet_ids)}")
    if security_group_ids:
        print(f"Security Groups: {', '.join(security_group_ids)}")
    
    mount_targets = manager.ensure_mount_targets(
        subnet_ids=subnet_ids,
        security_group_ids=security_group_ids
    )
    
    print(f"\n=== Mount Targets ({len(mount_targets)}) ===")
    for mt in mount_targets:
        mt_id = mt.get("MountTargetId") if isinstance(mt, dict) else mt
        print(f"  - {mt_id}")


if __name__ == "__main__":
    main()

"""
Network configuration for deployment scripts.
"""
import argparse
from pathlib import Path
from typing import Dict, Optional

from deploy_common import load_aws_config

try:
    from scripts.network_discovery import NetworkDiscovery
except ImportError:
    from network_discovery import NetworkDiscovery


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


def parse_args():
    """
    Parse CLI arguments.
    
    :returns: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Print ECS network configuration")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env")
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    services_dir = args.services_dir or Path.cwd()
    aws_config = load_aws_config(services_dir)
    discovery = NetworkDiscovery(region=aws_config["AWS_REGION"], vpc_id=aws_config.get("VPC_ID"))
    config = get_network_config(aws_config, discovery)
    print(config or "None")


if __name__ == "__main__":
    main()

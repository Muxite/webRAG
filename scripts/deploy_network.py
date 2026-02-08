"""
Network configuration for deployment scripts.
"""
from typing import Dict, Optional

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

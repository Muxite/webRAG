"""
Enable VPC DNS resolution and support for service discovery.
"""
import boto3
import sys
from pathlib import Path

try:
    from scripts.deploy_common import load_aws_config
except ImportError:
    from deploy_common import load_aws_config


def main():
    """Enable VPC DNS."""
    services_dir = Path.cwd()
    if (services_dir / "services").exists():
        services_dir = services_dir / "services"
    
    aws_config = load_aws_config(services_dir)
    region = aws_config["AWS_REGION"]
    
    ec2 = boto3.client("ec2", region_name=region)
    ecs = boto3.client("ecs", region_name=region)
    
    print("=== Fixing VPC DNS Configuration ===\n")
    
    # Get VPC from ECS service
    cluster = aws_config["ECS_CLUSTER"]
    response = ecs.describe_services(cluster=cluster, services=["euglena-gateway"])
    services = response.get("services", [])
    
    if not services:
        print("FAIL: Gateway service not found")
        sys.exit(1)
    
    net_config = services[0].get("networkConfiguration", {}).get("awsvpcConfiguration", {})
    subnets = net_config.get("subnets", [])
    
    if not subnets:
        print("FAIL: No subnets found")
        sys.exit(1)
    
    subnet = ec2.describe_subnets(SubnetIds=[subnets[0]])["Subnets"][0]
    vpc_id = subnet["VpcId"]
    
    print(f"VPC ID: {vpc_id}")
    
    # Check current DNS settings
    vpc = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]
    dns_resolution = vpc.get("EnableDnsHostnames", {}).get("Value", False)
    dns_support = vpc.get("EnableDnsSupport", {}).get("Value", False)
    
    print(f"Current DNS Resolution: {dns_resolution}")
    print(f"Current DNS Support: {dns_support}")
    
    # Enable DNS if needed
    if not dns_resolution:
        print("\nEnabling DNS Resolution...")
        ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})
        print("OK: DNS Resolution enabled")
    else:
        print("OK: DNS Resolution already enabled")
    
    if not dns_support:
        print("\nEnabling DNS Support...")
        ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={"Value": True})
        print("OK: DNS Support enabled")
    else:
        print("OK: DNS Support already enabled")
    
    print("\n=== VPC DNS Configuration Complete ===")
    print("Note: Agent tasks may need to be restarted to pick up DNS changes")


if __name__ == "__main__":
    main()

"""
Network discovery utilities for AWS VPC, subnets, and security groups.

Can be run directly or imported as a module.
"""
import boto3
from typing import Dict, List, Optional
import sys


class NetworkDiscovery:
    """
    Discovers and manages AWS network resources (VPCs, subnets, security groups).
    
    :param region: AWS region name.
    :param vpc_id: Optional VPC ID to use (if None, will auto-discover).
    """
    
    def __init__(self, region: str, vpc_id: Optional[str] = None):
        self.region = region
        self.vpc_id = vpc_id
        self.ec2_client = boto3.client("ec2", region_name=region)
        self._vpc_cache: Optional[Dict] = None
    
    def get_default_vpc(self) -> Optional[Dict]:
        """
        Get the default VPC for the region.
        
        :returns: VPC dictionary or None if not found.
        """
        try:
            response = self.ec2_client.describe_vpcs(
                Filters=[{"Name": "isDefault", "Values": ["true"]}]
            )
            vpcs = response.get("Vpcs", [])
            if vpcs:
                return vpcs[0]
        except Exception as e:
            print(f"Error getting default VPC: {e}", file=sys.stderr)
        return None
    
    def find_vpc(self, vpc_name: Optional[str] = None) -> Optional[Dict]:
        """
        Find VPC by ID, name, or use default.
        
        :param vpc_name: Optional VPC name tag to search for.
        :returns: VPC dictionary or None if not found.
        """
        if self._vpc_cache:
            return self._vpc_cache
        
        if self.vpc_id:
            try:
                response = self.ec2_client.describe_vpcs(VpcIds=[self.vpc_id])
                vpcs = response.get("Vpcs", [])
                if vpcs:
                    self._vpc_cache = vpcs[0]
                    return self._vpc_cache
            except Exception as e:
                print(f"Error finding VPC by ID {self.vpc_id}: {e}", file=sys.stderr)
        
        if vpc_name:
            try:
                response = self.ec2_client.describe_vpcs(
                    Filters=[{"Name": "tag:Name", "Values": [vpc_name]}]
                )
                vpcs = response.get("Vpcs", [])
                if vpcs:
                    self._vpc_cache = vpcs[0]
                    return self._vpc_cache
            except Exception as e:
                print(f"Error finding VPC by name {vpc_name}: {e}", file=sys.stderr)
        
        default_vpc = self.get_default_vpc()
        if default_vpc:
            self._vpc_cache = default_vpc
            return default_vpc
        
        return None
    
    def find_subnets(self, vpc_id: str, subnet_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Find subnets in a VPC.
        
        :param vpc_id: VPC ID to search in.
        :param subnet_names: Optional list of subnet name tags to filter by.
        :returns: List of subnet dictionaries.
        """
        try:
            filters = [{"Name": "vpc-id", "Values": [vpc_id]}]
            if subnet_names:
                filters.append({"Name": "tag:Name", "Values": subnet_names})
            
            response = self.ec2_client.describe_subnets(Filters=filters)
            return response.get("Subnets", [])
        except Exception as e:
            print(f"Error finding subnets: {e}", file=sys.stderr)
            return []
    
    def find_security_groups(self, vpc_id: str, sg_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Find security groups in a VPC.
        
        :param vpc_id: VPC ID to search in.
        :param sg_names: Optional list of security group names to filter by.
        :returns: List of security group dictionaries.
        """
        try:
            filters = [{"Name": "vpc-id", "Values": [vpc_id]}]
            if sg_names:
                filters.append({"Name": "group-name", "Values": sg_names})
            
            response = self.ec2_client.describe_security_groups(Filters=filters)
            return response.get("SecurityGroups", [])
        except Exception as e:
            print(f"Error finding security groups: {e}", file=sys.stderr)
            return []
    
    def discover_all(self) -> Dict:
        """
        Discover complete network configuration.
        
        :returns: Dictionary with vpc, subnets, and security_groups.
        """
        vpc = self.find_vpc()
        if not vpc:
            return {"vpc": None, "subnets": [], "security_groups": []}
        
        vpc_id = vpc["VpcId"]
        subnets = self.find_subnets(vpc_id)
        security_groups = self.find_security_groups(vpc_id)
        
        return {
            "vpc": vpc,
            "subnets": subnets,
            "security_groups": security_groups
        }


def main():
    """
    Main entry point when run directly.
    Prints discovered network information.
    """
    import argparse
    from pathlib import Path
    from dotenv import dotenv_values
    
    parser = argparse.ArgumentParser(description="Discover AWS network resources")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--vpc-id", help="VPC ID to use")
    parser.add_argument("--vpc-name", help="VPC name tag to search for")
    parser.add_argument("--services-dir", type=Path, default=None,
                       help="Services directory containing aws.env (defaults to current directory)")
    
    args = parser.parse_args()
    
    services_dir = args.services_dir or Path.cwd()
    aws_env_path = services_dir / "aws.env"
    
    if aws_env_path.exists():
        aws_config = dict(dotenv_values(str(aws_env_path)))
        region = args.region or aws_config.get("AWS_REGION")
        vpc_id = args.vpc_id or aws_config.get("VPC_ID")
    else:
        if not args.region:
            print(f"Error: --region required or aws.env must exist in current directory ({aws_env_path})", file=sys.stderr)
            sys.exit(1)
        region = args.region
        vpc_id = args.vpc_id
    
    discovery = NetworkDiscovery(region=region, vpc_id=vpc_id)
    network_info = discovery.discover_all()
    
    print("=== Network Discovery Results ===")
    if network_info["vpc"]:
        vpc = network_info["vpc"]
        print(f"\nVPC: {vpc['VpcId']}")
        print(f"  CIDR: {vpc.get('CidrBlock', 'N/A')}")
        print(f"  State: {vpc.get('State', 'N/A')}")
        
        print(f"\nSubnets ({len(network_info['subnets'])}):")
        for subnet in network_info["subnets"]:
            name = next((tag["Value"] for tag in subnet.get("Tags", []) if tag["Key"] == "Name"), "N/A")
            print(f"  - {subnet['SubnetId']} ({name})")
            print(f"    CIDR: {subnet.get('CidrBlock', 'N/A')}, AZ: {subnet.get('AvailabilityZone', 'N/A')}")
        
        print(f"\nSecurity Groups ({len(network_info['security_groups'])}):")
        for sg in network_info["security_groups"]:
            print(f"  - {sg['GroupId']} ({sg.get('GroupName', 'N/A')})")
            print(f"    Description: {sg.get('Description', 'N/A')}")
    else:
        print("\nNo VPC found")


if __name__ == "__main__":
    main()

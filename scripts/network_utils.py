"""
Network validation utilities.

Uses NetworkDiscovery class for network resource discovery.
Can be run directly or imported as a module.
"""
import boto3
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

try:
    from scripts.network_discovery import NetworkDiscovery
except ImportError:
    from network_discovery import NetworkDiscovery


def check_port_allowed(rules: List[Dict], port: int, protocol: str = "tcp", source_sg_id: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Check if a port is allowed in security group rules.
    
    :param rules: List of security group rules.
    :param port: Port number to check.
    :param protocol: Protocol (tcp, udp, etc.).
    :param source_sg_id: Optional source security group ID to check for.
    :returns: Tuple of (allowed, list of matching rules).
    """
    allowed = False
    matching_rules = []
    
    for rule in rules:
        rule_protocol = rule.get("IpProtocol", "-1")
        if rule_protocol != "-1" and rule_protocol.lower() != protocol.lower():
            continue
        
        from_port = rule.get("FromPort")
        to_port = rule.get("ToPort")
        
        if rule_protocol == "-1":
            port_match = True
        elif from_port is None or to_port is None:
            continue
        else:
            port_match = from_port <= port <= to_port
        
        if not port_match:
            continue
        
        ip_ranges = rule.get("IpRanges", [])
        user_id_groups = rule.get("UserIdGroupPairs", [])
        
        if source_sg_id:
            for group_pair in user_id_groups:
                if group_pair.get("GroupId") == source_sg_id:
                    allowed = True
                    matching_rules.append(f"{rule_protocol}:{from_port}-{to_port} from SG {source_sg_id}")
                    break
        else:
            if ip_ranges or user_id_groups:
                allowed = True
                if ip_ranges:
                    matching_rules.append(f"{rule_protocol}:{from_port}-{to_port} from {[r.get('CidrIp') for r in ip_ranges]}")
                if user_id_groups:
                    matching_rules.append(f"{rule_protocol}:{from_port}-{to_port} from SG {[g.get('GroupId') for g in user_id_groups]}")
    
    return allowed, matching_rules


def check_internet_access(ec2_client, vpc_id: str, subnet_ids: List[str]) -> Tuple[bool, str]:
    """
    Check if subnets have internet access via NAT Gateway or Internet Gateway.
    
    :param ec2_client: Boto3 EC2 client.
    :param vpc_id: VPC ID.
    :param subnet_ids: List of subnet IDs.
    :returns: Tuple of (has_internet_access, description).
    """
    try:
        route_tables = ec2_client.describe_route_tables(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]}
            ]
        )
        
        has_igw = False
        has_nat = False
        
        for route_table in route_tables.get("RouteTables", []):
            for route in route_table.get("Routes", []):
                gateway_id = route.get("GatewayId", "")
                nat_gateway_id = route.get("NatGatewayId", "")
                destination = route.get("DestinationCidrBlock", "")
                
                if destination == "0.0.0.0/0":
                    if gateway_id and gateway_id.startswith("igw-"):
                        has_igw = True
                    if nat_gateway_id:
                        has_nat = True
        
        if has_igw:
            return True, "Internet Gateway configured"
        elif has_nat:
            return True, "NAT Gateway configured"
        else:
            return False, "No Internet Gateway or NAT Gateway found for 0.0.0.0/0 route"
    
    except Exception as e:
        return False, f"Error checking internet access: {e}"


def validate_network_configuration(aws_config: Dict):
    """
    Validate VPC, security groups, and network configuration.
    Uses NetworkDiscovery to find network resources.
    
    :param aws_config: AWS configuration dictionary.
    :returns: Tuple of (is_valid, issues_list).
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config.get("ECS_CLUSTER", "euglena-cluster")
    gateway_service = aws_config.get("GATEWAY_SERVICE_NAME", "euglena-gateway")
    agent_service = aws_config.get("ECS_SERVICE_NAME", "euglena-agent")
    
    ecs_client = boto3.client("ecs", region_name=region)
    ec2_client = boto3.client("ec2", region_name=region)
    
    issues = []
    
    print("\n=== Network Configuration Validation ===")
    
    try:
        gateway_response = ecs_client.describe_services(cluster=cluster, services=[gateway_service])
        agent_response = ecs_client.describe_services(cluster=cluster, services=[agent_service])
        
        gateway_svc = gateway_response.get("services", [{}])[0]
        agent_svc = agent_response.get("services", [{}])[0]
        
        gateway_network = gateway_svc.get("networkConfiguration", {}).get("awsvpcConfiguration", {})
        agent_network = agent_svc.get("networkConfiguration", {}).get("awsvpcConfiguration", {})
        
        gateway_sgs = gateway_network.get("securityGroups", [])
        agent_sgs = agent_network.get("securityGroups", [])
        gateway_subnets = gateway_network.get("subnets", [])
        agent_subnets = agent_network.get("subnets", [])
        
        print(f"\n1. VPC and Subnet Configuration...")
        
        if not gateway_subnets or not agent_subnets:
            issues.append("Missing subnet configuration")
            print("  FAIL: Subnets not configured")
            return False, issues
        
        subnet_info = {}
        for subnet_id in set(gateway_subnets + agent_subnets):
            try:
                response = ec2_client.describe_subnets(SubnetIds=[subnet_id])
                if response.get("Subnets"):
                    subnet = response["Subnets"][0]
                    subnet_info[subnet_id] = {
                        "vpc_id": subnet["VpcId"],
                        "cidr": subnet["CidrBlock"],
                        "az": subnet["AvailabilityZone"]
                    }
            except Exception as e:
                issues.append(f"Error getting subnet {subnet_id}: {e}")
                print(f"  FAIL: Error getting subnet {subnet_id}: {e}")
        
        gateway_vpcs = {subnet_info[s]["vpc_id"] for s in gateway_subnets if s in subnet_info}
        agent_vpcs = {subnet_info[s]["vpc_id"] for s in agent_subnets if s in subnet_info}
        
        if len(gateway_vpcs) > 1 or len(agent_vpcs) > 1:
            issues.append("Services span multiple VPCs")
            print("  FAIL: Services are in multiple VPCs")
        elif gateway_vpcs != agent_vpcs:
            issues.append("Gateway and agent are in different VPCs")
            print(f"  FAIL: VPC mismatch: Gateway={gateway_vpcs}, Agent={agent_vpcs}")
        else:
            vpc_id = list(gateway_vpcs)[0] if gateway_vpcs else None
            print(f"  OK: Both services in same VPC: {vpc_id}")
        
        if set(gateway_subnets) != set(agent_subnets):
            print("  WARN: Subnets differ")
            print(f"    Gateway: {gateway_subnets}")
            print(f"    Agent: {agent_subnets}")
        else:
            print(f"  OK: Services in same subnets")
        
        print(f"\n2. Security Group Configuration...")
        
        if not gateway_sgs or not agent_sgs:
            issues.append("Missing security group configuration")
            print("  FAIL: Security groups not configured")
            return False, issues
        
        all_sgs = list(set(gateway_sgs + agent_sgs))
        
        for sg_id in all_sgs:
            try:
                response = ec2_client.describe_security_groups(GroupIds=[sg_id])
                if not response.get("SecurityGroups"):
                    issues.append(f"Security group {sg_id} not found")
                    print(f"  FAIL: Security group {sg_id} not found")
                    continue
                
                sg = response["SecurityGroups"][0]
                sg_name = sg.get("GroupName", "unknown")
                ingress = sg.get("IpPermissions", [])
                egress = sg.get("IpPermissionsEgress", [])
                
                print(f"  Security Group {sg_id} ({sg_name}):")
                print(f"    Ingress rules: {len(ingress)}")
                print(f"    Egress rules: {len(egress)}")
                
                required_ports = {
                    "tcp": [6379, 5672, 8000, 8080, 8081, 8082],
                    "udp": [53]
                }
                
                missing_ingress = []
                missing_egress = []
                
                for protocol, ports in required_ports.items():
                    for port in ports:
                        allowed_ingress, _ = check_port_allowed(ingress, port, protocol, source_sg_id=sg_id)
                        allowed_egress, _ = check_port_allowed(egress, port, protocol)
                        
                        if not allowed_ingress:
                            missing_ingress.append(f"{port}/{protocol}")
                            print(f"    FAIL: Port {port}/{protocol} not allowed in ingress from same security group")
                        if not allowed_egress:
                            missing_egress.append(f"{port}/{protocol}")
                            print(f"    FAIL: Port {port}/{protocol} not allowed in egress")
                
                if missing_ingress:
                    issues.append(f"Security group {sg_id} missing ingress rules for: {', '.join(missing_ingress)}")
                if missing_egress:
                    issues.append(f"Security group {sg_id} missing egress rules for: {', '.join(missing_egress)}")
                
                if gateway_sgs == agent_sgs:
                    print(f"    OK: Gateway and agent share same security groups")
            except Exception as e:
                issues.append(f"Error checking security group {sg_id}: {e}")
                print(f"  FAIL: Error checking security group {sg_id}: {e}")
        
        print(f"\n3. Internet Access...")
        if gateway_vpcs:
            vpc_id = list(gateway_vpcs)[0]
            has_internet, desc = check_internet_access(ec2_client, vpc_id, list(set(gateway_subnets + agent_subnets)))
            if has_internet:
                print(f"  OK: {desc}")
            else:
                issues.append(f"No internet access: {desc}")
                print(f"  FAIL: {desc}")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Error validating network: {e}")
        print(f"  FAIL: Error: {e}")
        return False, issues


def fix_security_group_rules(aws_config: Dict) -> Tuple[bool, List[str]]:
    """
    Fix security group rules to allow required ports for service communication.
    
    :param aws_config: AWS configuration dictionary.
    :returns: Tuple of (success, messages).
    """
    region = aws_config["AWS_REGION"]
    cluster = aws_config.get("ECS_CLUSTER", "euglena-cluster")
    gateway_service = aws_config.get("GATEWAY_SERVICE_NAME", "euglena-gateway")
    
    ecs_client = boto3.client("ecs", region_name=region)
    ec2_client = boto3.client("ec2", region_name=region)
    
    messages = []
    
    try:
        gateway_response = ecs_client.describe_services(cluster=cluster, services=[gateway_service])
        gateway_svc = gateway_response.get("services", [{}])[0]
        gateway_network = gateway_svc.get("networkConfiguration", {}).get("awsvpcConfiguration", {})
        gateway_sgs = gateway_network.get("securityGroups", [])
        
        if not gateway_sgs:
            messages.append("No security groups found for gateway service")
            return False, messages
        
        sg_id = gateway_sgs[0]
        
        print(f"\n=== Fixing Security Group Rules ===")
        print(f"Security Group: {sg_id}")
        
        response = ec2_client.describe_security_groups(GroupIds=[sg_id])
        if not response.get("SecurityGroups"):
            messages.append(f"Security group {sg_id} not found")
            return False, messages
        
        sg = response["SecurityGroups"][0]
        existing_ingress = sg.get("IpPermissions", [])
        
        required_ingress = [
            {"port": 6379, "protocol": "tcp", "description": "Redis from same security group"},
            {"port": 5672, "protocol": "tcp", "description": "RabbitMQ from same security group"},
            {"port": 8000, "protocol": "tcp", "description": "Chroma from same security group"},
            {"port": 8080, "protocol": "tcp", "description": "Gateway from same security group"},
            {"port": 8081, "protocol": "tcp", "description": "Agent from same security group"},
            {"port": 8082, "protocol": "tcp", "description": "Metrics from same security group"},
            {"port": 53, "protocol": "udp", "description": "DNS from same security group"}
        ]
        
        rules_to_add = []
        for req in required_ingress:
            port = req["port"]
            protocol = req["protocol"]
            description = req["description"]
            
            already_exists = False
            for existing_rule in existing_ingress:
                if (existing_rule.get("IpProtocol") == protocol and
                    existing_rule.get("FromPort") == port and
                    existing_rule.get("ToPort") == port):
                    for pair in existing_rule.get("UserIdGroupPairs", []):
                        if pair.get("GroupId") == sg_id:
                            already_exists = True
                            break
                    if already_exists:
                        break
            
            if not already_exists:
                rule_dict = {
                    "IpProtocol": protocol,
                    "FromPort": port,
                    "ToPort": port,
                    "UserIdGroupPairs": [{"GroupId": sg_id}]
                }
                rules_to_add.append((rule_dict, description))
        
        if not rules_to_add:
            messages.append("All required rules already exist")
            print("  OK: All required security group rules already exist")
            return True, messages
        
        print(f"  Adding {len(rules_to_add)} ingress rules...")
        for rule_dict, description in rules_to_add:
            print(f"    - {rule_dict['IpProtocol'].upper()}:{rule_dict['FromPort']} ({description})")
        
        rules_for_api = [rule_dict for rule_dict, _ in rules_to_add]
        
        try:
            ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=rules_for_api
            )
            messages.append(f"Successfully added {len(rules_to_add)} ingress rules")
            print(f"  OK: Successfully added security group rules")
            return True, messages
        except Exception as e:
            error_msg = f"Failed to add security group rules: {e}"
            messages.append(error_msg)
            print(f"  FAIL: {error_msg}")
            return False, messages
            
    except Exception as e:
        error_msg = f"Error fixing security group rules: {e}"
        messages.append(error_msg)
        print(f"  FAIL: {error_msg}")
        return False, messages


def main():
    """
    Main entry point when run directly.
    """
    from dotenv import dotenv_values
    
    services_dir = Path.cwd()
    aws_env_path = services_dir / "aws.env"
    
    if not aws_env_path.exists():
        print(f"Error: aws.env not found in current directory ({services_dir})", file=sys.stderr)
        print("  Please run this script from the services/ directory", file=sys.stderr)
        sys.exit(1)
    
    aws_config = dict(dotenv_values(str(aws_env_path)))
    
    print("=== Network Validation ===")
    is_valid, issues = validate_network_configuration(aws_config)
    
    if is_valid:
        print("\nOK: Network configuration is valid")
        sys.exit(0)
    else:
        print(f"\nWARN: Found {len(issues)} issues")
        print("\nAttempting to fix security group rules...")
        success, messages = fix_security_group_rules(aws_config)
        if success:
            print("\nOK: Security group rules fixed")
            is_valid, issues = validate_network_configuration(aws_config)
            if is_valid:
                print("\nOK: Network configuration is now valid")
                sys.exit(0)
        
        print(f"\nFAIL: Network configuration has issues")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()

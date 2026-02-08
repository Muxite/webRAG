"""
Diagnose agent connectivity to gateway services in autoscale mode.
"""
import boto3
import sys
from pathlib import Path

try:
    from scripts.deploy_common import load_aws_config
except ImportError:
    from deploy_common import load_aws_config


def main():
    """Diagnose agent connectivity issues."""
    services_dir = Path.cwd()
    if (services_dir / "services").exists():
        services_dir = services_dir / "services"
    
    aws_config = load_aws_config(services_dir)
    region = aws_config["AWS_REGION"]
    cluster = aws_config["ECS_CLUSTER"]
    
    ecs = boto3.client("ecs", region_name=region)
    sd = boto3.client("servicediscovery", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)
    
    print("=== Agent Connectivity Diagnosis ===\n")
    
    # Check service discovery
    print("1. Service Discovery:")
    namespace_name = aws_config.get("SERVICE_DISCOVERY_NAMESPACE", "euglena.local")
    namespaces = sd.list_namespaces()["Namespaces"]
    namespace = next((n for n in namespaces if n.get("Name") == namespace_name), None)
    
    if not namespace:
        print(f"  FAIL: Namespace {namespace_name} not found")
        return
    
    namespace_id = namespace["Id"]
    print(f"  OK: Namespace {namespace_name} (ID: {namespace_id})")
    
    services = sd.list_services(Filters=[{"Name": "NAMESPACE_ID", "Values": [namespace_id]}])["Services"]
    gateway_svc = next((s for s in services if s.get("Name") == "euglena-gateway"), None)
    
    if not gateway_svc:
        print("  FAIL: Gateway service not found in namespace")
        return
    
    service_id = gateway_svc["Id"]
    print(f"  OK: Gateway service (ID: {service_id})")
    
    instances = sd.list_instances(ServiceId=service_id)["Instances"]
    if not instances:
        print("  FAIL: No gateway instances registered")
        return
    
    for inst in instances:
        attrs = inst.get("Attributes", {})
        ip = attrs.get("AWS_INSTANCE_IPV4", "N/A")
        print(f"  OK: Gateway instance registered: {ip}")
    
    # Check ECS services
    print("\n2. ECS Services:")
    response = ecs.describe_services(cluster=cluster, services=["euglena-gateway", "euglena-agent"])
    services_list = response["services"]
    
    gateway_svc_ecs = next((s for s in services_list if s["serviceName"] == "euglena-gateway"), None)
    agent_svc_ecs = next((s for s in services_list if s["serviceName"] == "euglena-agent"), None)
    
    if gateway_svc_ecs:
        net_config = gateway_svc_ecs.get("networkConfiguration", {}).get("awsvpcConfiguration", {})
        gateway_sg = net_config.get("securityGroups", [])
        gateway_subnets = net_config.get("subnets", [])
        print(f"  Gateway: SG={gateway_sg}, Subnets={len(gateway_subnets)}")
    
    if agent_svc_ecs:
        net_config = agent_svc_ecs.get("networkConfiguration", {}).get("awsvpcConfiguration", {})
        agent_sg = net_config.get("securityGroups", [])
        agent_subnets = net_config.get("subnets", [])
        print(f"  Agent: SG={agent_sg}, Subnets={len(agent_subnets)}")
        
        if gateway_sg and agent_sg:
            if gateway_sg == agent_sg:
                print("  OK: Both services use same security group")
            else:
                print("  WARN: Services use different security groups")
    
    # Check security group rules
    print("\n3. Security Group Rules:")
    if gateway_sg:
        sg_id = gateway_sg[0]
        sg = ec2.describe_security_groups(GroupIds=[sg_id])["SecurityGroups"][0]
        
        required_ports = [5672, 6379, 8000]
        for port in required_ports:
            has_rule = False
            for rule in sg.get("IpPermissions", []):
                if rule.get("FromPort") == port and rule.get("ToPort") == port:
                    for pair in rule.get("UserIdGroupPairs", []):
                        if pair.get("GroupId") == sg_id:
                            has_rule = True
                            break
            if has_rule:
                print(f"  OK: Port {port} allows ingress from same SG")
            else:
                print(f"  FAIL: Port {port} missing ingress rule from same SG")
    
    # Check VPC DNS
    print("\n4. VPC DNS Configuration:")
    if gateway_subnets:
        subnet = ec2.describe_subnets(SubnetIds=[gateway_subnets[0]])["Subnets"][0]
        vpc_id = subnet["VpcId"]
        
        dns_resolution_attr = ec2.describe_vpc_attribute(VpcId=vpc_id, Attribute="enableDnsHostnames")
        dns_support_attr = ec2.describe_vpc_attribute(VpcId=vpc_id, Attribute="enableDnsSupport")
        
        dns_resolution = dns_resolution_attr.get("EnableDnsHostnames", {}).get("Value", False)
        dns_support = dns_support_attr.get("EnableDnsSupport", {}).get("Value", False)
        
        print(f"  VPC: {vpc_id}")
        print(f"  DNS Resolution: {dns_resolution}")
        print(f"  DNS Support: {dns_support}")
        
        if not dns_resolution or not dns_support:
            print("  FAIL: VPC DNS not enabled - agent cannot resolve service discovery DNS")
        else:
            print("  OK: VPC DNS enabled")
    
    # Check agent task definition
    print("\n5. Agent Task Definition:")
    agent_tasks = ecs.list_tasks(cluster=cluster, serviceName="euglena-agent")["taskArns"]
    if agent_tasks:
        task = ecs.describe_tasks(cluster=cluster, tasks=[agent_tasks[0]])["tasks"][0]
        td_arn = task["taskDefinitionArn"]
        td = ecs.describe_task_definition(taskDefinition=td_arn)["taskDefinition"]
        
        env_vars = {}
        for container in td.get("containerDefinitions", []):
            if container["name"] == "agent":
                for env in container.get("environment", []):
                    env_vars[env["name"]] = env["value"]
        
        print(f"  RABBITMQ_URL: {env_vars.get('RABBITMQ_URL', 'NOT SET')}")
        print(f"  REDIS_URL: {env_vars.get('REDIS_URL', 'NOT SET')}")
        print(f"  CHROMA_URL: {env_vars.get('CHROMA_URL', 'NOT SET')}")
        
        expected_host = "euglena-gateway.euglena.local"
        for key in ["RABBITMQ_URL", "REDIS_URL", "CHROMA_URL"]:
            url = env_vars.get(key, "")
            if expected_host in url:
                print(f"  OK: {key} uses service discovery hostname")
            else:
                print(f"  FAIL: {key} does not use service discovery hostname")
    
    print("\n=== Diagnosis Complete ===")


if __name__ == "__main__":
    main()
